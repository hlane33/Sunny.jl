"""
    apply_op(ϕ::MPS, opname::String, sites, siteidx::Int) -> MPS

Apply a local operator to an MPS (Matrix Product State) at a specific site.

# Arguments
- `ϕ::MPS`: The initial MPS state
- `opname::String`: Name of the operator to apply (e.g., "Sz")
- `sites`: The site indices defining the Hilbert space
- `siteidx::Int`: The specific site where the operator is applied

# Returns
- A new MPS with the operator applied at the specified site

# Notes
- Makes a copy of the original state before modification
- The state is orthogonalized at the target site before operator application
"""
function apply_op(ϕ::MPS, opname::String, sites, siteidx::Int)
    ϕ = copy(ϕ) # Make a copy of the original state
    orthogonalize!(ϕ, siteidx)
    new_ϕj = op(opname, sites[siteidx]) * ϕ[siteidx]
    noprime!(new_ϕj)
    ϕ[siteidx] = new_ϕj
    return ϕ
end

"""
    compute_G(N, ψ, ϕ, H, sites, η, ts, tstep, cutoff, maxdim) -> Matrix{ComplexF64}

Compute the time-dependent correlation function G(j,t) = ⟨ψ|Sⱼᶻ(t)|ϕ⟩e^{-ηt}/π.

# Arguments
- `N`: Number of sites
- `ψ`: Initial bra state (MPS)
- `ϕ`: Initial ket state (MPS)
- `H`: Hamiltonian (MPO)
- `sites`: Site indices
- `η`: Damping factor
- `ts`: Time points
- `tstep`: Time step size
- `cutoff`: SVD cutoff for TDVP
- `maxdim`: Maximum bond dimension for TDVP

# Returns
- `G`: Matrix of size N × length(ts) containing correlation values

# Notes
- Uses time-dependent variational principle (TDVP) for time evolution
- Normalizes states after each time step
"""
function compute_G(N, ψ, ϕ, H, sites, η, ts, tstep, cutoff, maxdim)
    G = Array{ComplexF64}(undef, N, length(ts))
    
    # Initial state measurements
    for j ∈ 1:N
        Sjz_ϕ = apply_op(ϕ, "Sz", sites, j)
        G[j, 1] = inner(ψ, Sjz_ϕ) * exp(-η * 0.0)/π
    end
    
    # Time evolution
    for (ti, t) in enumerate(ts[2:end])
        # Evolve both states using TDVP
        ϕ = tdvp(H, -tstep*im/2, ϕ;
                time_step=-tstep*im/2,
                nsteps=1,
                maxdim, 
                cutoff,
                outputlevel=0)
        
        ψ = tdvp(H, -tstep*im/2, ψ;
                time_step=-tstep*im/2,
                nsteps=1,
                maxdim, 
                cutoff,
                outputlevel=0)
        
        normalize!(ϕ)
        normalize!(ψ)
        
        # Measurements
        for j ∈ 1:N
            Sjz_ϕ = apply_op(ϕ, "Sz", sites, j)
            corr = inner(ψ, Sjz_ϕ) * exp(-η * t)/π
            G[j, ti+1] = corr
        end
        println("finished t = $t")
    end
    return G
end

"""
    linear_predict(y::Union{Vector{Float64}, Vector{ComplexF64}}; n_predict::Int, n_coeff::Int) -> Vector

Perform linear prediction to extend a time series using an autoregressive model.

# Arguments
- `y`: Input time series (real or complex)
- `n_predict`: Number of future points to predict
- `n_coeff`: Number of previous coefficients to use in the AR model

# Returns
- Extended time series (original + predicted points)

# Notes
- Uses least squares to fit the AR coefficients
- The time series must be longer than n_coeff
- For complex inputs, performs prediction in complex space
"""
function linear_predict(y::Union{Vector{Float64}, Vector{ComplexF64}}; n_predict::Int, n_coeff::Int)
    #n_coeff dictates how many previous values to use for prediction
    #n_predict is how many future values to predict
    n = length(y)
    n ≤ n_coeff && error("Time series too short for prediction.")
    
    # Construct the Toeplitz matrix for least-squares fitting
    Y = zeros(ComplexF64, n - n_coeff, n_coeff)
    for i in 1:(n - n_coeff)
        Y[i, :] = y[(n_coeff + i - 1):-1:i]
    end
    
    # Solve Y * d ≈ y[n_coeff+1:end] for coefficients d
    d = Y \ y[n_coeff+1:end]
    
    # Extrapolate using the AR model
    y_pred = copy(y)
    for i in (n+1):(n + n_predict)
        next_val = sum(d .* y_pred[(i-1):-1:(i - n_coeff)])
        push!(y_pred, next_val)
    end
    
    return y_pred
end

"""
    compute_S(G, qs, ωs, positions, c, ts; linear_predict_params) -> Matrix{Float64}

Compute the dynamical structure factor S(q,ω) using time-space Fourier transform.

# Arguments
- `G`: Correlation function matrix from `compute_G`
- `qs`: Momentum values (1D array)
- `ωs`: Energy values (1D array)
- `positions`: Spatial positions of sites
- `c`: Center site index
- `ts`: Time points
- `linear_predict_params`: Named tuple with `n_predict` and `n_coeff` for linear prediction

# Returns
- 2D array of S(q,ω) values

# Notes
- Uses linear prediction to extend time series if specified
- Implements the form from https://doi.org/10.1103/PhysRevB.107.165146
- Assumes a translationally invariant system
"""
function compute_S(G, qs, ωs, positions, c, ts; linear_predict_params)
    
    out = zeros(Float64, length(qs), length(ωs))

    # Extend time axis via linear prediction
    if linear_predict_params.n_predict > 0
        extended_ts = [ts; ts[end] .+ (1:linear_predict_params.n_predict) * (ts[2] - ts[1])]
        extended_G = zeros(ComplexF64, size(G,1), length(extended_ts))
        # Extrapolate each spatial point's time series
        for xi in 1:size(G,1)
            extended_G[xi, :] = linear_predict(G[xi, :]; linear_predict_params...)
        end
        ts = extended_ts
        G = extended_G
    end

    for (qi, q) ∈ enumerate(qs)
        for (ωi, ω) ∈ enumerate(ωs)
            sum_val = 0.0
            for xi ∈ 1:length(positions), ti ∈ 1:length(ts)
                val = cos(q * (positions[xi]-c)) * 
                      (cos(ω * ts[ti]) * real(G[xi, ti]) - 
                       sin(ω * ts[ti]) * imag(G[xi, ti]))
                sum_val += val
            end
            out[qi, ωi] = real(sum_val)
        end
    end
    return out
end

"""
    compute_S_v2(G, qs, ωs, positions, c, ts; linear_predict_params) -> Matrix{ComplexF64}

Compute the dynamical structure factor S(q,ω) using complex Fourier transform.

# Arguments
- `G`: Correlation function matrix from `compute_G`
- `qs`: Momentum values (1D array)
- `ωs`: Energy values (1D array)
- `positions`: Spatial positions of sites
- `c`: Center site index
- `ts`: Time points
- `linear_predict_params`: Named tuple with `n_predict` and `n_coeff` for linear prediction

# Returns
- 2D complex array of S(q,ω) values

# Notes
- Uses complex exponential for Fourier transform
- This can introduce extra term in the structure factor compared to the real version
- Uses linear prediction to extend time series if specified
"""
function compute_S_v2(G, qs, ωs, positions, c, ts; linear_predict_params)
    out = zeros(ComplexF64, length(qs), length(ωs))
    
    # Extend time axis via linear prediction
    if linear_predict_params.n_predict > 0
        extended_ts = [ts; ts[end] .+ (1:linear_predict_params.n_predict) * (ts[2] - ts[1])]
        extended_G = zeros(ComplexF64, size(G,1), length(extended_ts))
        # Extrapolate each spatial point's time series
        for xi in 1:size(G,1)
            extended_G[xi, :] = linear_predict(G[xi, :]; linear_predict_params...)
        end
        ts = extended_ts
        G = extended_G
    end

    # Compute Fourier transform on extended data
    for (qi, q) in enumerate(qs)
        for (ωi, ω) in enumerate(ωs)
            sum_val = 0.0
            for xi in 1:length(positions), ti in 1:length(ts)
                pos_factor = exp(im * q * (positions[xi] - c))
                time_factor = exp(im * ω * ts[ti])
                sum_val += pos_factor * time_factor * G[xi, ti]
            end
            out[qi, ωi] = sum_val
        end
    end

    println("Applied linear prediction with n_predict=$(linear_predict_params.n_predict)")
    return out
end