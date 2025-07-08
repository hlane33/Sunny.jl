using ITensors, ITensorMPS, Sunny, GLMakie

# Function to visualize the square lattice with GLMakie
function visualize_square_lattice(Nx::Int, Ny::Int; yperiodic=false)
    # Generate the lattice
    lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
    
    # Create figure
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1, 1], 
              title="$(Nx)×$(Ny) Square Lattice (yperiodic=$yperiodic)",
              aspect=DataAspect())
    
    # Calculate site coordinates
    site_coords = [(div(n-1, Ny)+1, mod(n-1, Ny)+1) for n in 1:Nx*Ny]
    
    # Plot sites
    scatter!(ax, Point2f.(site_coords), color=:blue, markersize=30)
    
    # Add site number labels
    for (i, (x, y)) in enumerate(site_coords)
        text!(ax, "$i", position=Point2f(x, y-0.15), align=(:center, :center), fontsize=20)
    end
    
    # Plot bonds
    for bond in lattice
        lines!(ax, [Point2f(bond.x1, bond.y1), Point2f(bond.x2, bond.y2)], 
               color=:black, linewidth=3)
    end
    
    # Adjust axis limits
    xlims!(ax, 0.5, Nx+0.5)
    ylims!(ax, 0.5, Ny+0.5)
    
    # Hide axis decorations for cleaner look
    hidespines!(ax)
    hidedecorations!(ax)
    
    display(fig)
    return fig
end
#Original calculation:


let
    Nx = 8
    Ny = 8
    N = Nx * Ny
    yperiodic = true

    # Initialize the site degrees of freedom.
    sites = siteinds("S=1/2", N; conserve_qns=true)

    # Use the AutoMPO feature to create the 
    # next-neighbor Heisenberg model.
    ops = OpSum()
    lattice = triangular_lattice(Nx, Ny; yperiodic=yperiodic)
    num_bonds = length(lattice)
    print(lattice)
    println("Number of bonds in the $Nx×$Ny lattice with yperiodic=$yperiodic: $num_bonds")


    # square lattice also available:
    # lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
    for bnd in lattice
        ops += 0.5, "S+", bnd.s1, "S-", bnd.s2
        ops += 0.5, "S-", bnd.s1, "S+", bnd.s2
        ops += "Sz", bnd.s1, "Sz", bnd.s2
    end
    H = MPO(ops, sites)

    # Set the initial wavefunction matrix product state
    # to be a Neel state.
    #
    # This choice implicitly sets the global Sz quantum number
    # of the wavefunction to zero. Since it is an MPS
    # it will remain in this quantum number sector.
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    psi0 = MPS(sites,state)


    # inner calculates matrix elements of MPO's with respect to MPS's
    # inner(psi0, H, psi0) = <psi0|H|psi0>
    println("ITENSOR CALCULATION ==============================")
    println("Periodic BC in y is", yperiodic)
    println("Initial energy = ", inner(psi0, Apply(H, psi0)))

    # Set the parameters controlling the accuracy of the DMRG
    # calculation for each DMRG sweep. 
    nsweeps = 10
    maxdim = [10, 20, 100, 100, 200]
    cutoff = [1E-8]
    noise = 1E-7,1E-8,0.0
    # Begin the DMRG calculation
    energy, psi = dmrg(H, psi0;nsweeps, maxdim, cutoff, noise)

    # Print the final energy reported by DMRG
    println("\nGround State Energy = ", energy)
    println("\nUsing overlap = ", inner(psi, Apply(H, psi)))

    println("\nTotal QN of Ground State = ", totalqn(psi))

    return
end