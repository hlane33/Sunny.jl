using Sunny: MeasureSpec, num_correlations, num_observables
mutable struct QuantumCorrelations
    # 𝒮^{αβ}(q,ω) data and metadata
    const data           :: Array{ComplexF64, 7}                 # Raw SF with sublattice indices (ncorrs × natoms × natoms × sys_dims × nω)
    const crystal        :: Crystal                              # Crystal for interpretation of q indices in `data`
    const origin_crystal :: Union{Nothing,Crystal}               # Original user-specified crystal (if different from above)
    const Δω             :: Float64                              # Energy step size 

    # Observable information
    measure            :: MeasureSpec                            # Storehouse for combiner. Mutable so combiner can be changed.
    const observables  # :: Array{Op, 5}                         # (nobs × npos x latsize) -- note change of ordering relative to MeasureSpec. TODO: determine type strategy
    const positions    :: Array{Vec3, 4}                         # Position of each operator in fractional coordinates (latsize x npos)
    const atom_idcs    :: Array{Int64, 4}                        # Atom index corresponding to position of observable.
    const corr_pairs   :: Vector{NTuple{2, Int}}                 # (ncorr)

    nsamples           :: Int64                                  # Number of accumulated samples (single number saved as array for mutability)

    # Buffers and precomputed data 
    const samplebuf    :: Array{ComplexF64, 6}                   # Buffer for observables (nobservables × sys_dims × natoms × nsnapshots)
    const corrbuf      :: Array{ComplexF64, 4}                   # Buffer for correlations (sys_dims × nω)
    const space_fft!   :: FFTW.AbstractFFTs.Plan                 # Pre-planned lattice FFT for samplebuf
    const time_fft!    :: FFTW.AbstractFFTs.Plan                 # Pre-planned time FFT for samplebuf
    const corr_fft!    :: FFTW.AbstractFFTs.Plan                 # Pre-planned time FFT for corrbuf 
    const corr_ifft!   :: FFTW.AbstractFFTs.Plan                 # Pre-planned time IFFT for corrbuf 
end

function Base.getproperty(qc::QuantumCorrelations, sym::Symbol)
    return sym == :sys_dims ? size(qc.samplebuf)[2:4] : getfield(qc, sym)
end

function Base.setproperty!(qc::QuantumCorrelations, sym::Symbol, val)
    if sym == :measure
        @assert qc.measure.observables ≈ val.observables "New MeasureSpec must contain identical observables."
        @assert all(x -> x == 1, qc.measure.corr_pairs .== val.corr_pairs) "New MeasureSpec must contain identical correlation pairs."
        setfield!(qc, :measure, val)
    else
        setfield!(qc, sym, val)
    end
end

"""
    clone_correlations(qc::QuantumCorrelations)

Create a copy of a `QuantumCorrelations`.
"""
function clone_correlations(qc::QuantumCorrelations)
    dims = size(qc.data)[2:4]
    # Avoid copies/deep copies of C-generated data structures
    space_fft! = 1/√prod(dims) * FFTW.plan_fft!(qc.samplebuf, (2,3,4))
    time_fft! = FFTW.plan_fft!(qc.samplebuf, 6)
    corr_fft! = FFTW.plan_fft!(qc.corrbuf, 4)
    corr_ifft! = FFTW.plan_ifft!(qc.corrbuf, 4)
    M = isnothing(qc.M) ? nothing : copy(qc.M)
    return QuantumCorrelations(
        copy(qc.data), qc.crystal, qc.origin_crystal, qc.Δω,
        deepcopy(qc.measure), copy(qc.observables), copy(qc.positions), copy(qc.atom_idcs), copy(qc.corr_pairs),
        qc.nsamples,
        copy(qc.samplebuf), copy(qc.corrbuf), space_fft!, time_fft!, corr_fft!, corr_ifft!
    )
end


function to_reshaped_rlu(qc::QuantumCorrelations, q)
    orig_cryst = @something qc.origin_crystal qc.crystal
    return qc.crystal.recipvecs \ orig_cryst.recipvecs * q
end

# Determine a step size and down sampling factor that results in precise
# satisfaction of user-specified energy values.
function adjusted_dt_and_downsampling_factor(dt, nω, ωmax)
    @assert π/dt > ωmax "Desired `ωmax` not possible with specified `dt`. Choose smaller `dt` value."

    # Assume nω is the number of non-negative frequencies and determine total
    # number of frequency bins.
    n_all_ω = 2(Int64(nω) - 1)

    # Find downsampling factor for the given `dt` that yields an `ωmax` higher
    # than or equal to given `ωmax`. Then adjust `dt` down so that specified
    # `ωmax` is satisfied exactly.
    Δω = ωmax/(nω-1)
    measperiod = ceil(Int, π/(dt * ωmax))
    dt_new = 2π/(Δω*measperiod*n_all_ω)

    # Warn the user if `dt` required drastic adjustment, which will slow
    # simulations.
    # if dt_new/dt < 0.9
    #     @warn "To satisify specified energy values, the step size adjusted down by more than 10% from a value of dt=$dt to dt=$dt_new"
    # end

    return dt_new, measperiod
end


"""
    QuantumCorrelations(sys::System, ts::LenStepRange, n_predict::Int, n_coeff::Int;
                             measure=nothing, energies=range(0, 5, length(ts)),
                             positions=nothing)

An object to accumulate samples of dynamical pair correlations from a preexisting G obect of quantum trajectories
"""
function QuantumCorrelations(sys::System, qs_length::Int, energies_length::Int;
                             measure=nothing, initial_energies=NaN,
                             positions=nothing)         
    #given that we now use manual FT instead of FFT, zero padding is not needed
    #just have this rather than the longer way in SampledCorrelations.jl
    n_all_ω = length(initial_energies) # Number of non-negative frequencies, including
    Δω = initial_energies[2] - initial_energies[1] # Energy step size

    # Determine the positions of the observables in the MeasureSpec. By default,
    # these will just be the atom indices.
    positions = if isnothing(positions)
        map(eachsite(sys)) do site
            sys.crystal.positions[site.I[4]]
        end
    else
        positions
    end

    # Determine the number of positions. For an unentangled system, this will
    # just be the number of atoms.
    npos = size(positions, 4) 

    # Determine which atom index is used to derive information about a given
    # physical position. This becomes relevant for entangled units. 
    atom_idcs = map(site -> site.I[4], eachsite(sys))

    measure = isnothing(measure) ? ssf_trace(sys) : measure
    num_observables(measure)
    samplebuf = zeros(ComplexF64, num_observables(measure), sys.dims..., npos, n_all_ω)
    corrbuf = zeros(ComplexF64, sys.dims..., n_all_ω)

    # The output data has n_all_ω frequencies and data accounts for extended data via linear prediction
    #Assumes 1D system
    data = zeros(ComplexF64, num_correlations(measure), npos, npos, qs_length, 1, 1, energies_length)

    # The normalization is defined so that the prod(sys.dims)-many estimates of
    # the structure factor produced by the correlation conj(space_fft!) *
    # space_fft! are correctly averaged over. The corresponding time-average
    # can't be applied in the same way because the number of estimates varies
    # with Δt. These conventions ensure consistency with this spec:
    # https://sunnysuite.github.io/Sunny.jl/dev/structure-factor.html
    space_fft! = 1/√prod(sys.dims) * FFTW.plan_fft!(samplebuf, (2,3,4))
    time_fft!  = FFTW.plan_fft!(samplebuf, 6)
    corr_fft!  = FFTW.plan_fft!(corrbuf, 4)
    corr_ifft! = FFTW.plan_ifft!(corrbuf, 4)

    # Initialize nsamples to zero. Make an array so can update dynamically
    # without making struct mutable.
    nsamples = 0 

    # Make Structure factor and add an initial sample
    origin_crystal = isnothing(sys.origin) ? nothing : sys.origin.crystal
    qc = QuantumCorrelations(data, sys.crystal, origin_crystal, Δω,
                             measure, copy(measure.observables), positions, atom_idcs, copy(measure.corr_pairs),
                             nsamples,
                             samplebuf, corrbuf, space_fft!, time_fft!, corr_fft!, corr_ifft!)

    return qc
end

function Base.show(io::IO, ::QuantumCorrelations)
    print(io, "QuantumCorrelations")
    # TODO: Add correlation info?
end

function Base.show(io::IO, ::MIME"text/plain", qc::QuantumCorrelations)
    (; crystal, nsamples) = qc
    nω = round(Int, size(qc.data)[7]/2)
    sys_dims = size(qc.data[4:6])
    printstyled(io, "QuantumCorrelations"; bold=true, color=:underline)
    println(io," ($(Base.format_bytes(Base.summarysize(qc))))")
    print(io,"[")
    printstyled(io,"S(q,ω)"; bold=true)
    print(io," | nω = $nω, Δω = $(round(qc.Δω, digits=4))")
    println(io," | $nsamples $(nsamples > 1 ? "samples" : "sample")]")
    println(io,"Lattice: $sys_dims × $(natoms(crystal))")
end

