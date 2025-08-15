using ITensors, ITensorMPS, GLMakie, Sunny
include("../ITensors_integration.jl")


# This is my attempt at the DMRG/TEBD implementation of this paper https://arxiv.org/pdf/2507.19412v1
# I think that the actual DMRG part of this paper does not require the system to 'know' that it is CuGeO_3
# but I could be wrong on this. I have tried to set up a J1-J2-δ model like they do in the paper but 
# it is failing when the symmetry wants to overrwrite the bonds. I think it will take someone more familiar
# with what sunny is doing to implement this properly but there is no reason I can think of why it wouldn't work
# with the integration, except perhaps that compute_S may fail with having multiple atoms per unit cell.
# This is currently unintegrated as it makes it easier to test, but see `../examples/AFM_chain.jl` for how this 
# would be implemented.


#################
# Core Functions for TEBD but TPVD could be used#
#################

function apply_op(ϕ::MPS, opname::String, sites, siteidx::Int)
    ϕ = copy(ϕ) # Make a copy of the original state
    orthogonalize!(ϕ, siteidx)
    new_ϕj = op(opname, sites[siteidx]) * ϕ[siteidx]
    noprime!(new_ϕj)
    ϕ[siteidx] = new_ϕj
    return ϕ
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


#############
# Sunny Set up #
############
function create_dimer_chain(Lx::Int, Ly::Int=1, Lz::Int=1;
                            a::Float64=3.5, s::Float64=0.5,
                            periodic_bc::Bool=false)

    # --- Paper parameters (meV) ---
    J1  = 13.79
    δ   = 0.04
    J1_strong = J1 * (1 + δ)   # intra-cell A→B
    J1_weak   = J1 * (1 - δ)   # inter-cell B→A across +x cell
    J2        = 0.35 * J1      # next-nearest neighbor A→A (and B→B)

    # --- Lattice geometry ---
    latvecs = lattice_vectors(a, 20.0, 20.0, 90, 90, 90)   # 1D along x
    positions = [[0.0, 0.0, 0.0],   # A
                 [0.5, 0.0, 0.0]]   # B
    crystal = Crystal(latvecs, position)

    # --- System in dipole mode ---
    sys = System(crystal, [1 => Moment(; s=s, g=2)], :dipole)

    view_crystal(sys)

    # --- Bond definitions ---
    nn_bond_strong = Bond(1, 2, [0, 0, 0])    # A→B intra-cell
    nn_bond_weak   = Bond(2, 1, [1, 0, 0])    # B→A in next cell
    nnn_bond_A     = Bond(1, 1, [1, 0, 0])    # A→A next cell
    nnn_bond_B     = Bond(2, 2, [1, 0, 0])    # B→B next cell

    # --- Set exchanges ---
    set_exchange!(sys, J1_strong, nn_bond_strong)
    set_exchange!(sys, J1_weak,   nn_bond_weak)
    set_exchange!(sys, J2,        nnn_bond_A)
    set_exchange!(sys, J2,        nnn_bond_B)

    # --- Build finite/periodic system ---
    sys = repeat_periodically(sys, (Lx, Ly, Lz))
    sys_inhom = to_inhomogeneous(sys)
    pbc = (periodic_bc, false, false)   # Only x may be periodic
    remove_periodicity!(sys_inhom, pbc)

    return sys_inhom
end



################
# Main Program #
################

function main()
    #Lattice size
    N = 15
    # Parameters for both DMRG and TEBD
    #DMRG
    nsweeps = 15
    maxdim = [10, 20, 100, 100, 200]
    #BOTH
    cutoff = 1E-10
    #TDVP
    η = 0.1
    tstep = 0.5
    tmax = 10.0

    # Run DMRG
    sys = create_dimer_chain(40, periodic_bc=false)
    DMRG_results, _ = calculate_ground_state(sys)
    ψ = DMRG_results.psi
    E0 = DMRG_results.energy
    sites = DMRG_results.sites
    println("Ground state energy: $E0")
    println("Sz expectation: ", expect(ψ, "Sz"))


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
              title = "S=1/2 CuGeO_3 dimerized spin chain N = $N")
    Makie.heatmap!(ax, allowed_qs, energies, out,
             colorrange = (0, 0.5 * maximum(out)))
    ylims!(ax, 0, 5)
    return fig
end

# Execute the program
fig = main()
display(fig)