using Sunny: MeasureSpec, num_correlations, num_observables


#######
#This file mimics Sunny.SampledCorrelations but creates an alternate QuantumCorrelations Object that 
# adjusts for quantum correlations rather than classical
######

"""
    QuantumCorrelations

Container for dynamical structure factor data 𝒮^{αβ}(q,ω) computed from quantum trajectories.

# Fields
- `data::Array{ComplexF64,7}`: Raw structure factor (ncorrs × natoms × natoms × sys_dims × nω)
- `crystal::Crystal`: Crystal structure for q-space interpretation
- `origin_crystal::Union{Nothing,Crystal}`: Original crystal before any transformations
- `Δω::Float64`: Energy step size
- `measure::MeasureSpec`: Measurement specification (mutable for combiner updates)
- `observables`: Operator definitions (nobs × npos × latsize)
- `positions::Array{Vec3,4}`: Fractional coordinates of operators
- `atom_idcs::Array{Int64,4}`: Atom indices for observable positions
- `corr_pairs::Vector{NTuple{2,Int}}`: Correlation component pairs
- `nsamples::Int64`: Accumulated sample count
- `samplebuf::Array{ComplexF64,6}`: Observable buffer (nobservables × sys_dims × natoms × nsnapshots)
- `corrbuf::Array{ComplexF64,4}`: Correlation buffer (sys_dims × nω)
"""
mutable struct QuantumCorrelations
    # 𝒮^{αβ}(q,ω) data and metadata
    const data           :: Array{ComplexF64, 7}                 # Raw SF with sublattice indices (ncorrs × natoms × natoms × sys_dims × nω)
    const system         :: System                               # System used for qc object
    const crystal        :: Crystal                              # Crystal for interpretation of q indices in `data`
    const origin_crystal :: Union{Nothing,Crystal}               # Original user-specified crystal (if different from above)

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
    to_reshaped_rlu(qc::QuantumCorrelations, q)

Convert q-vector from original to reshaped reciprocal lattice units.
"""
function to_reshaped_rlu(qc::QuantumCorrelations, q)
    orig_cryst = @something qc.origin_crystal qc.crystal
    return qc.crystal.recipvecs \ orig_cryst.recipvecs * q
end

"""
    QuantumCorrelations(sys::System, qs_length::Int, energies_length::Int; measure=nothing, initial_energies=NaN, positions=nothing)

Construct a QuantumCorrelations container for accumulating dynamical correlations.

# Arguments
- `sys::System`: Target spin system
- `qs_length::Int`: Number of q-points of external grid (qs)
- `energies_length::Int`: Number of energy bins of external applied energy grid (ωs)

# Keywords
- `measure=nothing`: Measurement specification (default: `ssf_trace(sys)`)
- `num_timesteps`: Number of timesteps used to sample G
- `positions=nothing`: Custom operator positions (default: crystal positions)

# Returns
- Preallocated `QuantumCorrelations` object with zero-initialized buffers

# Notes
- Uses manual Fourier transform (no zero-padding needed)
- Currently the only measure that will work is `ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false`
- Assumes 1D system for calculation of G (1D snaking of sites) so if you try a 2D system, and assign 
all the G[j,t] to just the first spatial dimension of samplebuf, it won't be long enough.  
"""
function QuantumCorrelations(sys::System, energies_length::Int,qxs_length::Int, qys_length::Int=1;
                             measure=ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                             num_timesteps::Int,
                             positions=nothing)         

    #given that we now use manual FT instead of FFT, zero padding is not needed
    #just have this rather than the longer way in SampledCorrelations.jl
    n_all_ω = num_timesteps #just makes clear the transform from time to energy steps

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
    data = zeros(ComplexF64, num_correlations(measure), npos, npos, qxs_length, qys_length, 1, energies_length)

    # Initialize nsamples to zero. Make an array so can update dynamically
    # without making struct mutable.
    nsamples = 0 

    # Make Structure factor and add an initial sample
    origin_crystal = isnothing(sys.origin) ? nothing : sys.origin.crystal
    qc = QuantumCorrelations(data, sys, sys.crystal, origin_crystal, 
                             measure, copy(measure.observables), positions, atom_idcs, copy(measure.corr_pairs),
                             nsamples,
                             samplebuf, corrbuf)

    return qc
end

function Base.show(io::IO, ::QuantumCorrelations)
    print(io, "QuantumCorrelations")
end

"""
    show(io::IO, ::MIME"text/plain", qc::QuantumCorrelations)

Display formatted summary of QuantumCorrelations object.

# Output Includes
- Memory usage
- Energy resolution (Δω)
- Sample count
- Lattice dimensions
"""
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
    println(io,"Lattice: $sys_dims × $(Sunny.natoms(crystal))")
end

