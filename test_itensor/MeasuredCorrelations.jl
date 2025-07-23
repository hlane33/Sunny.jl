using Sunny: MeasureSpec

"""
QuantumCorrelations compatible with Sunny.intensities using ssf_custom measure
"""
mutable struct QuantumCorrelations
    # Core data matching SampledCorrelations structure
    const data           :: Array{ComplexF64, 6}                 # (ncorrs, natoms, natoms, Lx, Ly, Lz) - static correlations 
    const M              :: Int                                  # Number of samples (always 1 for quantum)
    const crystal        :: Crystal                              # Crystal structure
    const origin_crystal :: Crystal                              # Original crystal before symmetry operations
    
    # Measurement configuration (matching SampledCorrelations)
    const measure        :: MeasureSpec                          # Measurement specification
    const samplebuf      :: Array{ComplexF64, 7}                # (M, ncorrs, natoms, natoms, Lx, Ly, Lz)
    
    # System information
    const sys_dims       :: Tuple{Int,Int,Int}                   # System dimensions (Lx, Ly, Lz)
    const natoms         :: Int                                  # Number of atoms per unit cell
    const nsites         :: Int                                  # Total number of sites
    
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
Constructor that processes quantum correlation data for ssf_custom((q, ssf) -> real(ssf[3, 3])) measure
"""
function QuantumCorrelations(G::Matrix{ComplexF64}, ts::Vector{Float64}, η::Float64, 
                           sys::System)
    # Extract system properties
    crystal = sys.crystal
    sys_dims = sys.dims
    natoms_per_cell = length(crystal.positions)
    
    # Verify dimensions
    nsites, ntimes = size(G)
    expected_sites = prod(sys_dims) * natoms_per_cell
    @assert nsites == expected_sites "Number of sites in G ($nsites) doesn't match system dimensions and atoms per cell ($expected_sites)"
    
    # Create the ssf_custom measure
    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys; apply_g=false)
    
    # Compute frequencies
    dt = ts[2] - ts[1]
    ωs_raw = fftfreq(ntimes, 1/dt) .* 2π
    energies = fftshift(ωs_raw)
    
    # Process correlations
    samplebuf, data = process_quantum_correlations_for_ssf(
        G, ts, η, sys_dims, natoms_per_cell
    )
    
    return QuantumCorrelations(
        data, 1, crystal, crystal, measure, samplebuf,
        sys_dims, natoms_per_cell, nsites,
        energies, dt, copy(G), copy(ts), η, 1
    )
end


"""
Process quantum correlation data to fill the 6 correlation pairs needed by ssf_custom
The key insight: your G(i,t) represents Sz-Sz correlations, which is corr_pair (3,3)
"""
function process_quantum_correlations_for_ssf(G::Array{ComplexF64,2}, ts::Vector{Float64}, η::Float64,
                                             sys_dims::Tuple{Int,Int,Int}, natoms::Int)
    nsites, ntimes = size(G)
    ncorrs = 6  # Number of correlation pairs in ssf_custom
    
    # Reshape G to spatial structure: (Lx, Ly, Lz, natoms, ntimes)
    G_spatial = reshape(G, sys_dims..., natoms, ntimes)
    
    # Apply exponential damping window
    window = exp.(-η .* ts)
    dt = ts[2] - ts[1]
    
    # Initialize samplebuf: (M=1, ncorrs=6, natoms, natoms, Lx, Ly, Lz)
    samplebuf = zeros(ComplexF64, 1, ncorrs, natoms, natoms, sys_dims...)
    
    # Fill correlation data
    # Since your G represents Sz correlations from TDVP, we map it to the (3,3) correlation pair
    for i in 1:sys_dims[1], j in 1:sys_dims[2], k in 1:sys_dims[3]
        for α in 1:natoms, β in 1:natoms
            
            if α == β
                # Diagonal correlations - use your G data for Sz-Sz (pair index 1 corresponds to (3,3))
                # Take time-integrated (static) value
                sz_sz_correlation = sum(G_spatial[i, j, k, α, :] .* window) * dt
                samplebuf[1, 1, α, β, i, j, k] = sz_sz_correlation  # (3,3) -> index 1
                
                # For other diagonal correlations, set to zero or estimate if you have the data
                # You could potentially estimate Sx-Sx and Sy-Sy from Sz-Sz using spin algebra
                samplebuf[1, 4, α, β, i, j, k] = 0.0  # Sy-Sy (2,2) -> index 4  
                samplebuf[1, 6, α, β, i, j, k] = 0.0  # Sx-Sx (1,1) -> index 6
                
                # Off-diagonal auto-correlations (imaginary parts typically)
                samplebuf[1, 2, α, β, i, j, k] = 0.0  # Sy-Sz (2,3) -> index 2
                samplebuf[1, 3, α, β, i, j, k] = 0.0  # Sx-Sz (1,3) -> index 3
                samplebuf[1, 5, α, β, i, j, k] = 0.0  # Sx-Sy (1,2) -> index 5
            else
                # Off-diagonal (inter-site) correlations
                # If your G contains cross-correlations, extract them here
                # For now, set to zero as placeholder
                for corr_idx in 1:ncorrs
                    samplebuf[1, corr_idx, α, β, i, j, k] = 0.0
                end
            end
        end
    end
    
    # Create static correlation data (time-integrated)
    data = copy(samplebuf[1, :, :, :, :, :, :])  # Remove sample dimension
    
    return samplebuf, data
end

"""
Alternative constructor for cases where you have full spin correlation data
"""
function QuantumCorrelations(G_full::Dict{String, Array{ComplexF64,2}}, ts::Vector{Float64}, η::Float64,
                           crystal::Crystal, sys_dims::Tuple{Int,Int,Int})
    
    # This version handles the case where you have separate G matrices for different spin components
    # G_full would contain keys like "xx", "yy", "zz", "xy", "xz", "yz" for all correlation pairs
    
    nsites, ntimes = size(G_full["zz"])  # Assuming "zz" exists
    natoms_per_cell = length(crystal.positions)
    
    measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false)
    
    dt = ts[2] - ts[1]
    ωs_raw = fftfreq(ntimes, 1/dt) .* 2π
    energies = fftshift(ωs_raw)
    
    samplebuf, data = process_full_spin_correlations_for_ssf(
        G_full, ts, η, sys_dims, natoms_per_cell
    )
    
    return QuantumCorrelations(
        data, 1, crystal, crystal, measure, samplebuf,
        sys_dims, natoms_per_cell, nsites,
        energies, dt, G_full["zz"], copy(ts), η, 1
    )
end

"""
Process full spin correlation data when all components are available
"""
function process_full_spin_correlations_for_ssf(G_full::Dict{String, Array{ComplexF64,2}}, 
                                               ts::Vector{Float64}, η::Float64,
                                               sys_dims::Tuple{Int,Int,Int}, natoms::Int)
    
    nsites, ntimes = size(G_full["zz"])
    ncorrs = 6
    dt = ts[2] - ts[1]
    window = exp.(-η .* ts)
    
    # Mapping from correlation pair indices to G_full keys
    corr_map = Dict(
        1 => "zz",  # (3,3) Sz-Sz
        2 => "yz",  # (2,3) Sy-Sz  
        3 => "xz",  # (1,3) Sx-Sz
        4 => "yy",  # (2,2) Sy-Sy
        5 => "xy",  # (1,2) Sx-Sy
        6 => "xx"   # (1,1) Sx-Sx
    )
    
    samplebuf = zeros(ComplexF64, 1, ncorrs, natoms, natoms, sys_dims...)
    
    for (corr_idx, key) in corr_map
        if haskey(G_full, key)
            G_spatial = reshape(G_full[key], sys_dims..., natoms, ntimes)
            
            for i in 1:sys_dims[1], j in 1:sys_dims[2], k in 1:sys_dims[3]
                for α in 1:natoms, β in 1:natoms
                    correlation = sum(G_spatial[i, j, k, α, :] .* window) * dt
                    samplebuf[1, corr_idx, α, β, i, j, k] = correlation
                end
            end
        end
    end
    
    data = copy(samplebuf[1, :, :, :, :, :, :])
    return samplebuf, data
end

# Interface compatibility methods
Base.getproperty(qc::QuantumCorrelations, sym::Symbol) = getfield(qc, sym)

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
    println(io,"Measure: real(SSF[3,3]) - Sz-Sz component")
end

