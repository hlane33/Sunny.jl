using ITensors, ITensorMPS, GLMakie, Sunny
include("sunny_toITensor.jl")
include("ITensor_to_Sunny.jl")

#################
# Core Functions #
#################

function apply_op(ϕ::MPS, opname::String, sites, siteidx::Int)
    ϕ = copy(ϕ) # Make a copy of the original state
    orthogonalize!(ϕ, siteidx)
    new_ϕj = op(opname, sites[siteidx]) * ϕ[siteidx]
    noprime!(new_ϕj)
    ϕ[siteidx] = new_ϕj
    return ϕ
end

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


################
# Main Program #
################

function Get_Structure_factor()
    units = Units(:meV, :angstrom)
    # Lattice configuration
    N = 15
    # Time evolution parameters
    η = 0.1
    tstep = 0.5
    tmax = 3.0
    cutoff = 1E-10
    maxdim = 300  

    # RUN DMRG from Sunny
    custom_dmrg_config = DMRGConfig(
        15,                     # nsweeps
        [10, 20, 100, 100, 200], # maxdim
        [1E-10],               # cutoff
        (0.0,)                 # noise
    )

    custom_config = LatticeConfig(CHAIN_1D, N, N, 1, 1.0, 1/2, 1.0, 0.0, 0.0, true)
    DMRG_results, sys = main_calculation(custom_config, custom_dmrg_config)
    ψ = DMRG_results.psi
    H = DMRG_results.H
    sites = DMRG_results.sites

    # Prepare time evolution
    ts = 0.0:tstep:tmax
    c = div(N, 2)
    ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

    # Compute correlation function using TDVP
    G = compute_G(N, ψ, ϕ, H, sites, η, collect(ts), tstep, cutoff, maxdim)

    # Compute structure factor

    energies = 0:0.05:5
    sc = Get_StructureFactor_with_Sunny(G, energies, sys)

    # Generate linearly spaced q-points
    qs = [[0, 0, 0], [1/2, 0, 0], [1/2, 1/2, 0], [0, 0, 0]]
    cryst = sys.crystal
    path = q_space_path(cryst, qs, 500)

    # 2. Compute S(q, ω)
    res = intensities(sc, path; energies, kT = nothing)
    

    # 3. Plot
    fig = plot_intensities(res; units, title="Intensities")

    
    return fig
end

# Execute the program
fig = Get_Structure_factor()
display(fig)