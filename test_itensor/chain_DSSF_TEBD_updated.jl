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

function DMRG_AFM(N, nsweeps, maxdim, cutoff; linkdims=10)
    sites = siteinds("S=1/2", N)
    print(sites)
    os = OpSum()
    for j = 1:N-1
        os += "Sz", j, "Sz", j+1
        os += 1/2, "S+", j, "S-", j+1
        os += 1/2, "S-", j, "S+", j+1
    end
    H = MPO(os, sites)
    psi0 = random_mps(sites; linkdims)
    E0, ψ = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    return E0, ψ, sites
end

function create_AFM_gates(N, tstep, sites, η)
    gates = ITensor[]
    for j in 1:(N - 1)
        s1 = sites[j]
        s2 = sites[j + 1]
        hj = op("Sz", s1) * op("Sz", s2) +
             1/2 * op("S+", s1) * op("S-", s2) +
             1/2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * tstep/2 * hj) * exp(-η * tstep)/π
        push!(gates, Gj)
    end
    append!(gates, reverse(gates))
    c = div(N, 2)
    return gates, c
end

function compute_G(N, ψ, ϕ, gates, sites, η, ts, cutoff)
    G = Array{ComplexF64}(undef, N, length(ts))
    for (ti, t) in enumerate(ts)
        ϕ = apply(gates, ϕ; cutoff)
        ψ = apply(gates, ψ; cutoff)
        normalize!(ϕ)
        normalize!(ψ)
        for j ∈ 1:N
            Sjz_ϕ = apply_op(ϕ, "Sz", sites, j)
            corr = inner(ψ, Sjz_ϕ) * exp(-η * t)/π
            G[j, ti] = corr
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

################
# Main Program #
################

function main()
    # Parameters
    N = 30
    nsweeps = 15
    maxdim = [10, 20, 100, 100, 200]
    cutoff = 1E-10
    η = 0.1
    tstep = 0.2
    tmax = 10.0

    # Run DMRG
    E0, ψ, sites = DMRG_AFM(N, nsweeps, maxdim, cutoff)
    println("Ground state energy: $E0")
    println("Sz expectation: ", expect(ψ, "Sz"))

    #RUN DMRG from Sunny
    N=30
    custom_dmrg_config = DMRGConfig(
    15,                     # nsweeps
    [10, 20, 100, 100, 200],# maxdim
    [1E-10],                   # cutoff
    (0.0,) # noise is empty
    )

    custom_chain_config = LatticeConfig(CHAIN_1D, N, 1, 1, 1.0, 1/2, 1.0, 0.0, 0.0, false)
    E0, ψ, H, sites, bond_pairs, coupling_groups = main_calculation(custom_chain_config, custom_dmrg_config)


    # Prepare time evolution
    ts = 0.0:tstep:tmax
    gates, c = create_AFM_gates(N, tstep, sites, η)
    ϕ = apply_op(ψ, "Sz", sites, c)  # Excited state

    # Compute correlation function
    G = compute_G(N, ψ, ϕ, gates, sites, η, ts, cutoff)

    # Compute structure factor
    energies = 0:0.05:5
    allowed_qs = 0:(1/N):2π
    positions = 1:N
    out = compute_S(allowed_qs, energies, G, positions, c, ts)

    # Plotting
    fig = Figure()
    ax = Axis(fig[1, 1],
              xlabel = "qₓ",
              xticks = ([0, allowed_qs[end]], ["0", "2π"]),
              ylabel = "Energy (meV)",
              title = "S=1/2 AFM DMRG/2nd order TEBD")
    heatmap!(ax, allowed_qs, energies, out,
             colorrange = (0, 0.5 * maximum(out)))
    ylims!(ax, 0, 5)
    return fig
end

# Execute the program
fig = main()
display(fig)