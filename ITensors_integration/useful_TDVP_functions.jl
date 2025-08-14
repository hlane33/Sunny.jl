######
#Contains useful functions for TDVP calculations
# 1) Structures to keep parameters for each section clean
# 2) Funcs to compute G[i,t]
# 3) Funcs to compute Fourier Transform to give S(q,ω)
# 4) Helpers for loading of G
######


# ----------------------
# Parameter structs
# ----------------------

"""
    TDVPParams

Structure to store parameters for Time-Dependent Variational Principle (TDVP) calculations.

# Fields
- `N::Int`: System size (number of sites).
- `η::Float64`: Damping factor or regularization parameter.
- `tstep::Float64`: Time step for the evolution.
- `tmax::Float64`: Maximum time for the evolution.
- `cutoff::Float64`: Singular value cutoff for truncation.
- `maxdim::Int`: Maximum bond dimension allowed.
- `periodic_bc::Bool`: Whether to use periodic boundary conditions (default: false).
"""
Base.@kwdef struct TDVPParams
    N::Int
    η::Float64
    tstep::Float64
    tmax::Float64
    cutoff::Float64
    maxdim::Int
    periodic_bc::Bool = false
end

"""
    FTParams

Structure to store parameters for Fourier transform calculations.

# Fields
- `allowed_qxs::Vector{Float64}`: Allowed momentum values in x dir(q-vectors).
- `allowed_qys::Vector{Float64}`: Allowed momentum values in y dir(q-vectors).
- `energies::AbstractVector{Float64}`: Energy values for the spectrum.
- `positions::AbstractVector{Int}`: Lattice positions to consider.
- `c::Int`: Central site or reference position.
- `ts::AbstractVector{Float64}`: Time points for the time evolution data.
"""
Base.@kwdef struct FTParams
    allowed_qxs::Vector{Float64}
    allowed_qys::Vector{Float64} = [pi]
    c::Int
    energies::Vector{Float64}
    positions::Matrix{Float64}  # Real-space positions from Sunny system [3 x Nsites]
    ts::Vector{Float64}
end

"""
    LinearPredictParams

Structure to store parameters for linear prediction calculations.

# Fields
- `n_predict::Int`: Number of points to predict.
- `n_coeff::Int`: Number of coefficients to use in the prediction.
"""
Base.@kwdef struct LinearPredictParams
    n_predict::Int = 0
    n_coeff::Int = 0
end


######
#Correlation Calculations
#####
"""
    apply_op(ϕ::MPS, opname::String, sites, siteidx::Int) -> MPS

Apply a local operator to an MPS (Matrix Product State) at a specific site.

# Arguments
- `ϕ`: Initial MPS state
- `opname`: Name of the operator (e.g., "Sz", "Sx")
- `sites`: Site indices defining the Hilbert space
- `siteidx`: Target site index for operator application

# Returns
- New MPS with operator applied at specified site

# Notes
- Makes a deep copy of the original state
- Orthogonalizes at target site before application
- Preserves original state's normalization
"""
function apply_op(ϕ::MPS, opname::String, sites, siteidx::Int)
    ϕ = deepcopy(ϕ)
    orthogonalize!(ϕ, siteidx)
    new_ϕj = op(opname, sites[siteidx]) * ϕ[siteidx]
    noprime!(new_ϕj)
    ϕ[siteidx] = new_ϕj
    return ϕ
end

"""
    compute_G(tdvp_params, ft_params, ψ, ϕ, H, sites) -> Matrix{ComplexF64}

Compute time-dependent correlation function G(j,t) = ⟨ψ|Sⱼᶻ(t)|ϕ⟩e^{-ηt}/π.

# Arguments
- `tdvp_params`: TDVP parameters structure containing:
    - N: Number of lattice sites
    - η: Exponential damping factor
    - tstep: Time step for evolution
    - cutoff: SVD truncation cutoff
    - maxdim: Maximum bond dimension
- `ft_params`: Fourier transform parameters structure containing:
    - ts: Time points for evaluation
    - other FT-related parameters
- `ψ`: Initial bra state (MPS)
- `ϕ`: Initial ket state (MPS)
- `H`: System Hamiltonian (MPO)
- `sites`: Site collection

# Returns
- N × length(ts) matrix of complex correlation values

# Notes
- Uses TDVP for time evolution of both ψ and ϕ states
- Implements finite-time damping via η parameter
- Normalizes states after each evolution step
- Measurements are performed at all sites for each time point
- Progress is logged with @info for each completed time step

# Example
```julia
G = compute_G(tdvp_params, ft_params, ψ, ϕ, H, sites)
"""
function compute_G(tdvp_params::TDVPParams, ft_params::FTParams, ψ, ϕ, H, sites)
    N = tdvp_params.N
    η = tdvp_params.η
    ts = ft_params.ts
    tstep = tdvp_params.tstep
    cutoff = tdvp_params.cutoff
    maxdim = tdvp_params.maxdim
    
    G = Matrix{ComplexF64}(undef, N, length(ts))
    
    # Initial measurements
    @views begin
        for j in 1:N
            Sjz_ϕ = apply_op(ϕ, "Sz", sites, j)
            G[j, 1] = inner(ψ, Sjz_ϕ) * exp(-η * first(ts))/π
        end
    end

    # Time evolution loop
    for (ti, t) in enumerate(ts[2:end])
        ϕ = tdvp(H, -tstep*im/2, ϕ; maxdim, cutoff, outputlevel=0)
        ψ = tdvp(H, -tstep*im/2, ψ; maxdim, cutoff, outputlevel=0)
        
        normalize!.( (ϕ, ψ) )  # Tuple broadcasting
        
        # Parallel measurement (Threads.@threads for actual parallelism)
        @views for j in 1:N
            Sjz_ϕ = apply_op(ϕ, "Sz", sites, j)
            G[j, ti+1] = inner(ψ, Sjz_ϕ) * exp(-η * t)/π
        end
        @info "Completed t = $t"
    end
    return G
end


###########
#Structure Factor Calculations
##########

"""
    linear_predict(y::AbstractVector{T}; n_predict::Int, n_coeff::Int) where {T<:Union{Float64,ComplexF64}}

Perform linear prediction using autoregressive modeling.

# Arguments
- `y`: Input time series (real or complex)
- `n_predict`: Number of future points to predict
- `n_coeff`: Autoregressive model order

# Returns
- Extended time series (original + predicted)

# Throws
- `ArgumentError` if input too short for prediction

# Notes
Using Linear predict does exacerbate the asymmetry found when using [`@ref compute_S_complex`](@ref)
"""
function linear_predict(y::AbstractVector{T}; n_predict::Int, n_coeff::Int) where {T<:Union{Float64,ComplexF64}}
    length(y) ≤ n_coeff && throw(ArgumentError("Time series shorter than AR order"))
    
    # Construct Toeplitz matrix
    Y = zeros(T, length(y) - n_coeff, n_coeff)
    for i in 1:size(Y,1)
        Y[i,:] = @view y[(n_coeff+i-1):-1:i]
    end
    
    # Solve AR coefficients
    d = Y \ @view y[n_coeff+1:end]
    
    # Extrapolate
    y_pred = copy(y)
    for i in (length(y)+1):(length(y)+n_predict)
        next_val = sum(d .* @view y_pred[(i-1):-1:(i-n_coeff)])
        push!(y_pred, next_val)
    end
    
    return y_pred
end

"""
    compute_S(G, ft_params; linear_predict_params) -> Matrix{Float64}

Compute S(q,ω) via real-space Fourier transform.

# Arguments
- `G`: Correlation matrix from `compute_G` (N × length(ts) complex matrix)
- `ft_params::FTParams`: Fourier transform parameters containing:
    - allowed_qs: Momentum values (Vector{Float64})
    - energies: Frequency values (AbstractVector{Float64})
    - positions: Spatial positions (AbstractVector{Int})
    - c: Central site index (Int)
    - ts: Time points (AbstractVector{Float64})

# Keywords
- `linear_predict_params::LinearPredictParams`: Parameters for time extension containing:
    - n_predict: Number of future points to predict
    - n_coeff: Autoregressive model order

# Returns
- Real-valued S(q,ω) matrix (length(qs) × length(ωs))

# Notes
- Performs cosine/sine transforms for real-valued output
- Optionally extends time series using linear prediction
- Uses precomputed trigonometric terms for efficiency
- Extension to 2D would require tracking positions of coordinates in 2D i.e qs becomes an array of 2D momenta
"""
function compute_S(G::AbstractMatrix{ComplexF64}, ft_params::FTParams; linear_predict_params::LinearPredictParams)
   

    qs = ft_params.allowed_qxs
    ωs = ft_params.energies
    positions_3D = ft_params.positions
    c = ft_params.c
    ts = ft_params.ts

    #convert positions to just 1D
    positions = positions_3D[1,:] .+ 1
    
    # Time series extension
    if linear_predict_params.n_predict > 0
        ts, G = _extend_time_series(ts, G; linear_predict_params)
    end

    # Precompute trigonometric terms
    cos_ωt = [cos(ω*t) for ω in ωs, t in ts]
    sin_ωt = [sin(ω*t) for ω in ωs, t in ts]
    cos_qr = [cos(q*(r-c)) for q in qs, r in positions]

    # Main computation
    out = zeros(Float64, length(qs), length(ωs))
    @inbounds for (qi,q) in enumerate(qs), (ωi,ω) in enumerate(ωs)
        sum_val = 0.0
        for xi in 1:length(positions), ti in 1:length(ts)
            sum_val += cos_qr[qi,xi] * (cos_ωt[ωi,ti] * real(G[xi,ti]) - sin_ωt[ωi,ti] * imag(G[xi,ti]))
        end
        out[qi,ωi] = sum_val
    end
    
    return out
end

"""
    compute_S_complex(G, ft_params; linear_predict_params) -> Matrix{ComplexF64}

Compute S(q,ω) via complex Fourier transform.

# Arguments
- `G`: Correlation matrix from `compute_G` (N × length(ts) complex matrix)
- `ft_params::FTParams`: Fourier transform parameters containing:
    - allowed_qs: Momentum values (Vector{Float64})
    - energies: Frequency values (AbstractVector{Float64})
    - positions: Spatial positions (AbstractVector{Int})
    - c: Central site index (Int)
    - ts: Time points (AbstractVector{Float64})

# Keywords
- `linear_predict_params::LinearPredictParams`: Parameters for time extension containing:
    - n_predict: Number of future points to predict
    - n_coeff: Autoregressive model order

# Returns
- Complex-valued S(q,ω) matrix (length(qs) × length(ωs))

# Notes
- Performs complex exponential transforms
- Optionally extends time series using linear prediction
- Only works for 1D chain system
- Preserves full phase information
"""
function compute_S_complex(G::AbstractMatrix{ComplexF64}, ft_params::FTParams; linear_predict_params::LinearPredictParams)
    qs = ft_params.allowed_qxs
    ωs = ft_params.energies
    positions = ft_params.positions
    c = ft_params.c
    ts = ft_params.ts

    
    #convert positions to just 1D (The plus one aligns with choice of c in TDPVP)
    positions = positions[1,:] .+ 1
    # Time series extension
    if linear_predict_params.n_predict > 0
        ts, G = _extend_time_series(ts, G; linear_predict_params)
    end

    # Precompute phase factors
    phase_qr = [exp(im*q*(r-c)) for q in qs, r in positions]
    phase_ωt = [exp(im*ω*t) for ω in ωs, t in ts]

    # Main computation
    out = zeros(ComplexF64, length(qs), length(ωs))
    @inbounds for (qi,q) in enumerate(qs), (ωi,ω) in enumerate(ωs)
        out[qi,ωi] = sum(phase_qr[qi,xi] * phase_ωt[ωi,ti] * G[xi,ti] 
                     for xi in 1:length(positions), ti in 1:length(ts))
    end

    @info "Applied linear prediction: n_predict=$(linear_predict_params.n_predict)"
    return out
end

# Helper function for time series extension
function _extend_time_series(ts, G; linear_predict_params::LinearPredictParams)
    dt = ts[2] - ts[1]
    extended_ts = [ts; ts[end] .+ (1:linear_predict_params.n_predict)*dt]
    extended_G = similar(G, size(G,1), length(extended_ts))
    
    Threads.@threads for xi in 1:size(G,1)
        extended_G[xi,:] = linear_predict(G[xi,:]; n_predict=linear_predict_params.n_predict,
                          n_coeff=linear_predict_params.n_coeff)
    end
    
    return extended_ts, extended_G
end

"""
    extract_positions_from_sunny(sys::System) -> Matrix{Float64}

Extract real-space positions of all sites from a Sunny system and return them as a 3×N matrix.

# Arguments
- `sys::System`: A Sunny system object containing crystal structure and site information.

# Returns
- `pos_matrix::Matrix{Float64}`: A 3×N matrix where each column represents the [x, y, z] coordinates 
  of a site in real space (Angstroms). N is the total number of sites in the system.

# Notes
- For 1D or 2D systems, the unused dimensions will be zero filled.
- Positions start at zero not one, so for a 1D chain pos_matrix[1,:] = [0.0, 1.0, ...]

"""
function extract_positions_from_sunny(sys)
    # Get all site positions from the Sunny system
    # sys.crystal.positions contains fractional coordinates
    # sys.dims gives the system dimensions
    
    positions = []
    for site in Sunny.eachsite(sys)
        # Get real-space position for this site
        r = Sunny.global_position(sys, site)  # Returns [x, y, z] in Angstroms
        push!(positions, r)
    end
    
    # Convert to matrix format [3 x Nsites] for easier indexing
    pos_matrix = hcat(positions...)
    return pos_matrix
end

#################
#Serialization and wrappers
#################

"""
    save_object(obj, filename)

Serialize and save object to file.

# Arguments
- `obj`: Julia object to serialize
- `filename`: Destination file path
"""
function save_object(obj, filename)
    open(filename, "w") do io
        serialize(io, obj)
    end
end

"""
    load_object(filename) -> Any

Load serialized object from file.

# Arguments
- `filename`: Source file path

# Returns
- Deserialized Julia object
"""
function load_object(filename)
    open(filename, "r") do io
        deserialize(io)
    end
end



"""
    map_G_to_2D(G::AbstractMatrix{ComplexF64}, sys::System, Lt::Int)

Maps a Quantum correlation array from a 1D compressed format to a 5D array representation 
that matches the system's real-space dimensions and basis structure.

# Arguments
- `sys::System`: The system object containing crystal and dimension information
- `Lt::Int`: The number of time slices (or other discrete values in the 5th dimension)

# Returns
- `G_2D::Array{ComplexF64,5}`: A 5-dimensional array with dimensions (Lx, Ly, Lz, N_basis, Lt)
  representing the correlation function in real space and orbital basis

# Note
- The function assumes the existence of global variables `G` (the input Green's function)
  and `n_to_cartind` (the index mapping dictionary)
- The input Green's function `G` should have dimensions (N, Lt) where N = Lx × Ly × Lz × N_basis
"""
function map_G_to_2D(G::AbstractMatrix{ComplexF64}, sys::System,Lt::Int)
    Lx, Ly, Lz = sys.dims
    N_basis = length(sys.crystal.positions)
    G_2D = zeros(ComplexF64, Lx, Ly, Lz, N_basis, Lt)

    for n in 1:size(G, 1)
        i, j, k, l = n_to_cartind[n]  # k=1 (2D), l=1 or 2
        G_2D[i,j,k,l,:] .= G[n,:]
    end

    return G_2D
end


"""
    compute_G_wrapper(sys, tdvp_params, ft_params, linear_predict_params; dmrg_config)

High-level wrapper for computing G matrix with DMRG initialization.

# Arguments
- `sys`: The system for which to compute the correlation function
- `tdvp_params`: TDVP parameters structure containing:
    - N: System size
    - η: Damping factor
    - tstep: Evolution time step
    - cutoff: Truncation cutoff
    - maxdim: Maximum bond dimension
- `ft_params`: Fourier transform parameters structure containing:
    - ts: Time points
    - c: Central site position
    - other FT-related parameters
- `linear_predict_params`: Linear prediction parameters (currently unused in this function)

# Keywords
- `dmrg_config`: DMRG configuration dictionary

# Returns
- Computed G matrix (N × length(ts) complex matrix)
- n_to_cartind (inverse mapping of n --> [i,j,k,l] see [`cartind_to_label`](@ref)

# Notes
- Uses TDVP time evolution to compute the dynamical correlation function
- The ϕ state is created by applying Sz operator at central site (ft_params.c)
"""
function compute_G_wrapper(sys, tdvp_params, ft_params, linear_predict_params; dmrg_config)
    DMRG_results, n_to_cartind = calculate_ground_state(sys; dmrg_config)
    ϕ = apply_op(DMRG_results.psi, "Sz", DMRG_results.sites, ft_params.c)
    G = compute_G(tdvp_params, ft_params, DMRG_results.psi, ϕ, DMRG_results.H, DMRG_results.sites)
    return (G, n_to_cartind)  # Return as a tuple to be saved
end

"""
    load_G(g_filename, compute_func, compute_args...; dmrg_config=default_dmrg_config())

Load or compute G matrix and n --> [i,j,k,l] site mapping with caching.

# Arguments
- `g_filename`: Cache file path
- `compute_func`: Function to compute G if not cached
- `compute_args`: Arguments for compute_func

# Keywords
- `dmrg_config`: DMRG configuration

# Returns
- G matrix (loaded or computed)
- 
"""
function load_G(g_filename, compute_func, compute_args...; dmrg_config=default_dmrg_config())
    if isfile(g_filename)
        @info "Loading cached G and n_to_cartind from $g_filename"
        return load_object(g_filename)  # Returns the tuple (G, n_to_cartind)
    else
        @info "Computing G matrix and n_to_cartind mapping..."
        result = compute_func(compute_args...; dmrg_config)  # This returns (G, n_to_cartind)
        save_object(result, g_filename)
        @info "Saved G and n_to_cartind to $g_filename"
        return result
    end
end

