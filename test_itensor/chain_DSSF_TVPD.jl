using ITensors, ITensorMPS, GLMakie, Sunny
include("sunny_toITensor.jl")

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

function compute_S(qs, ωs, G, positions, c, ts)
    out = zeros(Float64, length(qs), length(ωs))
    for (qi, q) ∈ enumerate(qs)
        for (ωi, ω) ∈ enumerate(ωs)
            sum_val = 0.0
            for xi ∈ 1:length(positions), ti ∈ 1:length(ts)
                val = cos(q * (positions[xi] - c)) * 
                      (cos(ω * ts[ti]) * real(G[xi, ti]) - 
                       sin(ω * ts[ti]) * imag(G[xi, ti]))
                sum_val += val
            end
            out[qi, ωi] = sum_val
        end
    end
    return out
end

function compute_S_low_res(G, positions, c, ts)
    out = zeros(Float64, length(positions), length(ts))
    
    for (qi, q_pos) ∈ enumerate(positions)
        for (ωi, ω_t) ∈ enumerate(ts)
            sum_val = 0.0
            for xi ∈ 1:length(positions), ti ∈ 1:length(ts)
                val = cos(q_pos * (positions[xi] - c)) *
                      (cos(ω_t * ts[ti]) * real(G[xi, ti]) -
                       sin(ω_t * ts[ti]) * imag(G[xi, ti]))
                sum_val += val
            end
            out[qi, ωi] = sum_val
        end
    end
    return out
end

################
# Main Program #
################

function main()
    # Parameters
    N = 15
    η = 0.1
    tstep = 0.5
    tmax = 10.0
    cutoff = 1E-10
    maxdim = 300  # For TDVP evolution

    # RUN DMRG from Sunny
    custom_dmrg_config = DMRGConfig(
        15,                     # nsweeps
        [10, 20, 100, 100, 200], # maxdim
        [1E-10],               # cutoff
        (0.0,)                 # noise
    )

    sys = create_chain_system(20; periodic_bc = false)
    DMRG_results = calculate_ground_state(sys)
    ψ = DMRG_results.psi
    H = DMRG_results.H
    sites = DMRG_results.sites

    # Prepare time evolution
    ts = 0.0:tstep:tmax
    c = div(N, 2)
    ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

    # Compute correlation function using TDVP
    G = compute_G(N, ψ, ϕ, H, sites, η, collect(ts), tstep, cutoff, maxdim)

    print("correlations", G)


    # Compute structure factor
    energies = 0:0.05:5
    allowed_qs = 0:(1/N):2π
    positions = 1:N
    out = compute_S(allowed_qs, energies, G, positions, c, ts)
    print("structure factor output: ", out)

    # Plotting
    fig = Figure()
    ax = Axis(fig[1, 1],
            xlabel = "qₓ",
            xticks = ([0, allowed_qs[end]], ["0", "2π"]),
            ylabel = "Energy (meV)",
            title = "S=1/2 AFM DMRG/TDVP for Chain lattice")

    # Create heatmap with controlled color range
    vmax = 0.5 * maximum(out)  # Set upper limit for better contrast
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


    return fig
end

# Execute the program
fig = main()
display(fig)