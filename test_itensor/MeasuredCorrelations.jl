using Sunny: MeasureSpec
using FFTW

"""
QuantumCorrelations compatible with Sunny
"""
mutable struct QuantumCorrelations
    # Core data matching SampledCorrelations structure
    const data           :: Array{ComplexF64, 7}                 # (ncorrs, natoms, natoms, Lx, Ly, Lz, nω) - static correlations 
    const M              :: Int                                  # Number of samples (always 1 for quantum)
    const crystal        :: Crystal                              # Crystal structure
    const origin_crystal :: Crystal                              # Original crystal before symmetry operations
    
    # Measurement configuration (matching SampledCorrelations)
    const Δω             :: Float64
    const measure        :: MeasureSpec                          # Measurement specification
    const samplebuf      :: Array{ComplexF64, 8}                # (M, ncorrs, natoms, natoms, Lx, Ly, Lz, nω)
    
    # System information
    const sys_dims       :: Tuple{Int,Int,Int}                   # System dimensions (Lx, Ly, Lz)
    const natoms         :: Int                                  # Number of atoms per unit cell
    const nsites         :: Int                                  # Total number of sites
    const positions      :: Array{Vec3, 4}                         # Position of each site in fractional coordinates
    
    # Energy/frequency information for dynamic correlations
    const energies       :: Vector{Float64}                     # Energy/frequency grid
    const Δt             :: Float64                              # Time step used in correlation computation
    
    # Quantum-specific metadata (for reference)
    const raw_correlations :: Array{ComplexF64, 2}               # Original G(i,t) data 
    const ts             :: Vector{Float64}                      # Time array used
    const η              :: Float64                              # Damping parameter
    
    # Sample tracking
    nsamples             :: Int                                  # Always 1 for quantum case
end

"""
Enhanced process_quantum_correlations_for_ssf that properly computes S(q,ω)
from TDVP trajectories G[site, time] -> S(q,ω) matching Sunny's data layout
"""
function process_quantum_correlations_for_ssf(G::Array{ComplexF64,2}, ts::Vector{Float64}, η::Float64,
                                             sys_dims::Tuple{Int,Int,Int}, natoms::Int)
    nsites, ntimes = size(G)
    ncorrs = 6  # Number of correlation pairs in ssf_custom
    
    println("Processing G with size: $(size(G))")
    println("System dimensions: $sys_dims, natoms: $natoms")
    
    # Verify dimensions
    expected_sites = prod(sys_dims) * natoms
    @assert nsites == expected_sites "Sites mismatch: got $nsites, expected $expected_sites"
    
    # Reshape G to spatial structure: (Lx, Ly, Lz, natoms, ntimes)
    G_spatial = reshape(G, sys_dims..., natoms, ntimes)
    
    # Time step and frequency setup
    dt = ts[2] - ts[1]
    ωs_raw = fftfreq(ntimes, 1/dt) .* 2π
    n_all_ω = ntimes
    
    # Create damping window
    window = exp.(-η .* ts)
    
    println("Frequency grid: $(length(ωs_raw)) points, dt = $dt")
    println("Applied damping with η = $η")
    
    # Initialize arrays to match Sunny's expected structure
    # data: (ncorrs, natoms, natoms, Lx, Ly, Lz, nω) - full S(q,ω) data
    # samplebuf: (M=1, ncorrs, natoms, natoms, Lx, Ly, Lz, nω) - same but with sample dimension
    data = zeros(ComplexF64, ncorrs, natoms, natoms, sys_dims..., n_all_ω)
    samplebuf = zeros(ComplexF64, 1, ncorrs, natoms, natoms, sys_dims..., n_all_ω)
    
    println("Computing temporal FFT for each site...")
    
    # Step 1: Temporal FFT for each real-space site
    G_ω = zeros(ComplexF64, sys_dims..., natoms, n_all_ω)
    for i in 1:sys_dims[1], j in 1:sys_dims[2], k in 1:sys_dims[3]
        for α in 1:natoms
            # Apply damping and temporal FFT
            correlation_t = G_spatial[i, j, k, α, :] .* window
            G_ω[i, j, k, α, :] = fft(correlation_t) * dt
        end
    end
    
    println("Computing spatial FFT to get S(q,ω)...")
    
    # Step 2: Spatial FFT to get S(q,ω) for each frequency
    for ω_idx in 1:n_all_ω
        for α in 1:natoms
            # Extract real-space data for this (α, ω)
            real_space_slice = G_ω[:, :, :, α, ω_idx]
            
            # Apply 3D spatial FFT to get q-space with proper normalization
            q_space_slice = fft(real_space_slice) / prod(sys_dims)
            
            # Store in data array - only diagonal (α,α) correlations for Sz-Sz
            # Index 1 corresponds to (3,3) component in ssf_custom measure
            data[1, α, α, :, :, :, ω_idx] = q_space_slice
            
            # Set other correlation components to zero for now
            for corr_idx in 2:ncorrs
                data[corr_idx, α, α, :, :, :, ω_idx] .= 0.0
            end
            
            # Off-diagonal atom correlations set to zero
            for β in 1:natoms
                if β != α
                    for corr_idx in 1:ncorrs
                        data[corr_idx, α, β, :, :, :, ω_idx] .= 0.0
                    end
                end
            end
        end
    end
    
    # Fill samplebuf with single sample
    samplebuf[1, :, :, :, :, :, :, :] = data
    
    println("Finished processing. Data shape: $(size(data))")
    println("Sample data range: [$(minimum(real(data))), $(maximum(real(data)))]")
    
    return samplebuf, data, ωs_raw
end

"""
Updated QuantumCorrelations constructor with proper structure factor computation
"""
function QuantumCorrelations(G::Matrix{ComplexF64}, ts::Vector{Float64}, η::Float64, 
                           sys::System; positions=nothing)
    # Extract system properties
    crystal = sys.crystal
    sys_dims = sys.dims
    natoms_per_cell = length(crystal.positions)
    
    println("Creating QuantumCorrelations object...")
    println("System dimensions: $sys_dims")
    println("Atoms per cell: $natoms_per_cell")
    
    # Verify dimensions
    nsites, ntimes = size(G)
    expected_sites = prod(sys_dims) * natoms_per_cell
    @assert nsites == expected_sites "Number of sites in G ($nsites) doesn't match system dimensions and atoms per cell ($expected_sites)"
    
    # Fixed positions handling using the provided code
    positions = if isnothing(positions)
        map(eachsite(sys)) do site
            sys.crystal.positions[site.I[4]]
        end
    else
        positions
    end
    print("POsitions type:", typeof(positions))
    # Create the ssf_custom measure for Sz-Sz correlations
    measure = ssf_custom((q, ssf) -> real(ssf[1]), sys; apply_g=false)  # Fixed indexing
    
    # Compute frequencies - keep original order for processing
    dt = ts[2] - ts[1]
    ωs_raw = fftfreq(ntimes, 1/dt) .* 2π
    energies = copy(ωs_raw)  # Don't shift here, keep FFT order
    
    # Process correlations with proper FFTs
    samplebuf, data, ω_grid = process_quantum_correlations_for_ssf(
        G, ts, η, sys_dims, natoms_per_cell
    )
    
    println("samplebuf size: $(size(samplebuf))")
    
    # Calculate Δω for compatibility
    Δω = length(energies) > 1 ? abs(energies[2] - energies[1]) : NaN
    
    println("QuantumCorrelations object created successfully")
    println("Data shape: $(size(data))")
    println("Frequency resolution Δω: $Δω")
    
    return QuantumCorrelations(
        data,                    # Full S(q,ω) data: (ncorrs, natoms, natoms, Lx, Ly, Lz, nω)
        1,                       # M = 1 sample
        crystal,                 # Crystal structure
        crystal,                 # Origin crystal
        Δω,                      # Frequency resolution
        measure,                 # Measurement specification  
        samplebuf,               # Sample buffer (M, ncorrs, natoms, natoms, Lx, Ly, Lz, nω)
        sys_dims,                # System dimensions
        natoms_per_cell,         # Number of atoms per cell
        nsites,                  # Total sites
        positions,               # Positions vector (fixed)
        energies,                # Energy/frequency grid
        dt,                      # Time step
        copy(G),                 # Raw correlations
        copy(ts),                # Time array
        η,                       # Damping parameter
        1                        # Number of samples
    )
end

function Base.show(io::IO, ::MIME"text/plain", qc::QuantumCorrelations)
    nω = length(qc.energies)
    printstyled(io, "QuantumCorrelations"; bold=true, color=:underline)
    println(io," ($(Base.format_bytes(Base.summarysize(qc))))")
    print(io,"[")
    printstyled(io,"S(q,ω)"; bold=true)
    print(io," | $(qc.nsamples) sample | nω = $nω, Δt = $(round(qc.Δt, digits=4))")
    println(io," | quantum Sz-Sz correlations]")
    println(io,"Lattice: $(qc.sys_dims) × $(qc.natoms)")
    println(io,"Damping: η = $(qc.η)")
    println(io,"Measure: real(SSF[1]) - Sz-Sz component")
end

# Interface compatibility methods
Base.getproperty(qc::QuantumCorrelations, sym::Symbol) = getfield(qc, sym)