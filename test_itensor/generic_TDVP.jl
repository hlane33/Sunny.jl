using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny

#Decide where you want these to actually be included
include("sunny_toITensor.jl")
include("ITensor_to_Sunny.jl")
include("MeasuredCorrelations.jl")
include("overloaded_intensities.jl")
include("CorrelationMeasuring.jl")



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
    print("Size of G",size(G))
    return G
end


################
# Main Program #
################

function Get_Structure_factor()
    units = Units(:meV, :angstrom)
    # Lattice configuration
    N = 20
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

    
    sys = create_chain_system(N; periodic_bc = true)
    DMRG_results = calculate_ground_state(sys)
    ψ = DMRG_results.psi
    H = DMRG_results.H
    sites = DMRG_results.sites

    # Prepare time evolution
    ts = 0.0:tstep:tmax
    N_timesteps = size(ts,1)
    c = div(N, 2)
    ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

    # Compute correlation function using TDVP
    G = compute_G(N, ψ, ϕ, H, sites, η, collect(ts), tstep, cutoff, maxdim)
    energies = range(0, 5, N_timesteps) #No. Energies has to match N timesteps so that data sizing for 

    # Generate linearly spaced q-points and intensities params
    cryst = sys.crystal
    qs = [[0,0,0], [1,0,0]]
    path = q_space_path(cryst, qs, 401)

    integrated = false #decides which method to do
    if integrated
        # Using SampledCorrelations Augmentation (INTEGRATED WAY)
        qc = Get_StructureFactor_with_Sunny(G, energies, sys)
        res = Sunny.intensities(qc, path; energies = :available, kT=nothing)
        #Julia not distinguishing between overloaded functions properly?!
        fig = plot_intensities(res; units, title="Dynamic structure factor for 1D chain Integrated", saturation=0.5)
    else
        # UNINTEGRATED WAY USING QuantumCorrelations
        qc = QuantumCorrelations(sys; 
                            measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                            energies=energies)

        add_sample!(qc,G)
        manual_plot = true # Set to true to plot manually
        if manual_plot
            # Extract G for a specific observable and other fixed indices
            obs_idx = 3  # or whichever observable you want
            # Assuming sys.dims = (Lx, Ly, ...) and you want to fix other spatial dimensions
            y_idx = 1    # fix y-coordinate if 2D/3D system
            z_idx = 1    # fix z-coordinate if 3D system  
            pos_idx = 1  # fix npos dimension

            # Extract the 2D slice: G[site, time]
            G_slice = G = qc.data[obs_idx, 1, 1, :, y_idx, z_idx, :]   # Shape: (Lx, n_all_ω)
            allowed_qs = 0:(1/N):2π
            out = real(G_slice) #compute_S(new_allowed_qs, energies, G_slice, positions, c, ts)

            # Plotting
            fig = Figure()
            ax = Axis(fig[1, 1],
                    xlabel = "qₓ",
                    xticks = ([0, allowed_qs[end]], ["0", "2π"]),
                    ylabel = "Energy (meV)",
                    title = "S=1/2 AFM DMRG/LLD for Chain lattice, manual plot")

            # Create heatmap with controlled color range
            vmax = 0.4 * maximum(out)  # Set upper limit for better contrast
            hm = heatmap!(ax, allowed_qs, energies, out,
                        colorrange = (0, vmax),
                        colormap = :viridis)  # :viridis is perceptually uniform

            # Add colorbar with clear labeling
            cbar = Colorbar(fig[1, 2], hm,
                        label = "Intensity (a.u.)",
                        vertical = true,
                        ticks = LinearTicks(5),
                        flipaxis = false)

            # Indicate clipped values in colorbar (optional)
            cbar.limits = (0, vmax)  # Explicitly show the range
            cbar.highclip = :red      # Color for values > vmax
            cbar.lowclip = :blue      # Color for values < 0 (if needed)

            # Set axis limits
            ylims!(ax, 0, 5)


        else
            res = intensities(qc, path; energies = :available, kT=nothing)
            fig = plot_intensities(res; units, title="Dynamic structure factor for 1D chain with qc", saturation=0.5)
        end
    end
    return fig

end

# Execute the program
fig = Get_Structure_factor()
display(fig)