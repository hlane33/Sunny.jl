using ITensors, ITensorMPS, GLMakie, FFTW
using Sunny, Serialization

#Decide where you want these to actually be included
include("sunny_toITensor.jl")
include("ITensor_to_Sunny.jl")
include("MeasuredCorrelations.jl")
include("overloaded_intensities.jl")
include("CorrelationMeasuring.jl")


#Serialization functions for saving G

function save_object(obj, filename)
    open(filename, "w") do io
        serialize(io, obj)
    end
end

function load_object(filename)
    open(filename, "r") do io
        deserialize(io)
    end
end

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
    tmax = 5.0
    ts = 0.0:tstep:tmax
    Lt = length(ts)
    cutoff = 1E-10
    maxdim = 300  

    #Create system 
    sys = create_chain_system(N; periodic_bc = false)
    cryst = sys.crystal
    q_ends = [[0,0,0], [1,0,0]]
    path = q_space_path(cryst, q_ends, 400)
    qpts = path.qs #qs in SVector 
    path_qs = [2pi*q[1] for q in qpts]
    c = div(N, 2)
    c_frac = (c - 1) / (N - 1)
    positions = 1:N
    positions_frac = (positions .- 1) ./ (N - 1)

    #Fourier transform params
    new_allowed_qs = (2pi/N) * (0:(N-1))
    allowed_qs = 0:(1/N):2π
    FT_params = (
        allowed_qs = path_qs,
        energies = range(0, 5, 500),
        positions = positions,
        c = c,
        ts = ts
    )
    #Linear prediciton params: n_predict is the number of future time steps to predict, 
    # n_coeff is the number of coefficients used in linear prediction
    linear_predict_params = (
        n_predict = 0,
        n_coeff = 0
    )


    
    # File to save/load G array
    g_filename = "G_array_$(N)sites_$(tmax)tmax.jls"

    # Try to load G if file exists
    if isfile(g_filename)
        println("Loading G array from $g_filename")
        G = load_object(g_filename) 
    else
        # Compute G if file doesn't exist
        custom_dmrg_config = DMRGConfig(
            15,                     # nsweeps
            [10, 20, 100, 100, 200], # maxdim
            [1E-10],               # cutoff
            (0.0,)                 # noise
        )
        DMRG_results = calculate_ground_state(sys)
        ψ = DMRG_results.psi
        H = DMRG_results.H
        sites = DMRG_results.sites
        ϕ = apply_op(ψ, "Sz", sites, FT_params.c)

        println("Computing G array...")
        G = compute_G(N, ψ, ϕ, H, sites, η, collect(ts), tstep, cutoff, maxdim)
        
        # Save the computed G array
        save_object(G, g_filename)
        println("Saved G array to $g_filename")
    end

    #If False: Intensities()
    manual_plot = false  

    # UNINTEGRATED WAY USING QuantumCorrelations
    qs_length = length(FT_params.allowed_qs)
    energies_length = length(FT_params.energies) #energies after FT 
    qc = QuantumCorrelations(sys, qs_length, energies_length ; 
                        measure = ssf_custom((q, ssf) -> real(ssf[3, 3]), sys;apply_g=false),
                        initial_energies = range(0, 5, length(ts))
                        )

    add_sample!(qc, G, FT_params, linear_predict_params)
   
    if manual_plot
        # Now independent plotting section
        # Extract the data we want to plot
        corr_idx = 1 #Correlation pair corresponding to SzSz in this case
        y_idx = 1  # fix y-coordinate if 1 system
        z_idx = 1  # fix z-coordinate if 2D system
        pos_idx = 1  #atom in unit cell
        data_slice = qc.data[corr_idx, pos_idx, pos_idx, :, y_idx, z_idx, :]
        plot_data = real(data_slice)
        # Create the figure
        fig = Figure()
        ax = Axis(fig[1, 1],
                xlabel = "qₓ",
                xticks = ([0, allowed_qs[end]], ["0", "2π"]),
                ylabel = "Energy (meV)",
                title = "S=1/2 AFM DMRG manual plot for Chain lattice")

        # Create heatmap
        vmax = 0.4 * maximum(plot_data)
        hm = heatmap!(ax, allowed_qs, FT_params.energies, plot_data,
                    colorrange = (0, vmax),
                    colormap = :viridis)

        # Add colorbar
        cbar = Colorbar(fig[1, 2], hm,
                    label = "Intensity (a.u.)",
                    vertical = true)

        ylims!(ax, 0, 5)
    else
        # Generate linearly spaced q-points and intensities params
        # Standard Sunny plotting with FT done in accum_sample! and plotting by plot_intensities
        res = intensities(qc, path)
        fig = plot_intensities(res; units, title="Dynamic structure factor for 1D chain with intensities()", saturation=0.9)
    end
return fig, res
end



# Execute the program
fig, res = Get_Structure_factor()
display(fig)