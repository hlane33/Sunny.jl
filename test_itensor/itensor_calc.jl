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



function plot_lattice(results::DMRGResults; show_crystal=false, coupling_threshold=1e-10)
    fig = Figure(resolution=(1200, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Get site positions and convert to Point2[]
    sites = get_site_positions(results.config, results.N_basis)
    points = [Point2(p[1], p[2]) for p in sites]

    # Plot sites
    scatter!(ax, points, color=:black, markersize=15)

    # Plot bonds if available
    if !isempty(results.bond_pairs)
        for (i, j, J) in results.bond_pairs
            if abs(J) > coupling_threshold
                lines!(ax, [points[i], points[j]], color=:blue, linewidth=1)
            end
        end
    end

    display(fig)
    return fig
end

"""
    honeycomb_lattice(Nx::Int,
                      Ny::Int;
                      kwargs...)::Lattice

Return a Lattice (array of LatticeBond objects) corresponding to the 
two-dimensional honeycomb lattice of dimensions (Nx,Ny).

The honeycomb lattice consists of two sublattices (A and B) arranged
in a hexagonal pattern. Each unit cell contains 2 sites, and each site
has exactly 3 nearest neighbors.

By default the lattice has open boundaries, but can be made periodic 
in the y direction by specifying the keyword argument `yperiodic=true`.

The lattice vectors are:
- a1 = (3/2, √3/2)  
- a2 = (3/2, -√3/2)

Arguments:
- Nx::Int: Number of unit cells in x direction
- Ny::Int: Number of unit cells in y direction  
- yperiodic::Bool: Whether to use periodic boundary conditions in y direction

Returns:
- Lattice: Vector of LatticeBond objects representing the honeycomb lattice
"""
function honeycomb_lattice(Nx::Int, Ny::Int; yperiodic=false)::Lattice
    yperiodic = yperiodic && (Ny > 2)
    
    # Total number of sites (2 per unit cell)
    N = 2 * Nx * Ny
    
    # Lattice constants
    a = 1.0  # nearest neighbor distance
    sqrt3 = sqrt(3.0)
    
    # Helper function to get site number from unit cell and sublattice
    function site_number(i::Int, j::Int, sublattice::Int)
        # i, j are unit cell coordinates (1-indexed)
        # sublattice: 1 for A, 2 for B
        return 2 * ((i - 1) * Ny + (j - 1)) + sublattice
    end
    
    # Helper function to get coordinates for a site
    function site_coords(i::Int, j::Int, sublattice::Int)
        # Unit cell position
        x_cell = (i - 1) * 3/2 * a
        y_cell = (j - 1) * sqrt3 * a
        
        # Sublattice offset
        if sublattice == 1  # A sublattice
            x_offset = 0.0
            y_offset = 0.0
        else  # B sublattice  
            x_offset = a
            y_offset = 0.0
        end
        
        return (x_cell + x_offset, y_cell + y_offset)
    end
    
    # Estimate number of bonds (3 bonds per site, but each bond counted twice)
    # Plus corrections for boundaries
    Nbond_estimate = 3 * N ÷ 2
    bonds = LatticeBond[]
    
    # Generate bonds
    for i in 1:Nx
        for j in 1:Ny
            # A site in current unit cell
            site_A = site_number(i, j, 1)
            x_A, y_A = site_coords(i, j, 1)
            
            # B site in current unit cell  
            site_B = site_number(i, j, 2)
            x_B, y_B = site_coords(i, j, 2)
            
            # Bond 1: A to B within same unit cell
            push!(bonds, LatticeBond(site_A, site_B, x_A, y_A, x_B, y_B, "intra"))
            
            # Bond 2: A to B in unit cell to the right (if exists)
            if i < Nx
                site_B_right = site_number(i + 1, j, 2)
                x_B_right, y_B_right = site_coords(i + 1, j, 2)
                push!(bonds, LatticeBond(site_A, site_B_right, x_A, y_A, x_B_right, y_B_right, "inter"))
            end
            
            # Bond 3: A to B in unit cell above (if exists or periodic)
            if j < Ny
                site_B_up = site_number(i, j + 1, 2)
                x_B_up, y_B_up = site_coords(i, j + 1, 2)
                push!(bonds, LatticeBond(site_A, site_B_up, x_A, y_A, x_B_up, y_B_up, "inter"))
            elseif yperiodic && j == Ny
                # Periodic boundary: connect to bottom
                site_B_periodic = site_number(i, 1, 2)
                x_B_periodic, y_B_periodic = site_coords(i, 1, 2)
                # Adjust y coordinate for periodic boundary
                y_B_periodic += Ny * sqrt3 * a
                push!(bonds, LatticeBond(site_A, site_B_periodic, x_A, y_A, x_B_periodic, y_B_periodic, "periodic"))
            end
        end
    end
    
    return bonds
end

# Additional helper functions for honeycomb lattice analysis

"""
    honeycomb_sublattice(site::Int)::Int

Return the sublattice index (1 for A, 2 for B) for a given site number
in a honeycomb lattice.
"""
function honeycomb_sublattice(site::Int)::Int
    return ((site - 1) % 2) + 1
end

"""
    honeycomb_unit_cell(site::Int, Ny::Int)::Tuple{Int,Int}

Return the unit cell coordinates (i,j) for a given site number
in a honeycomb lattice with Ny unit cells in the y direction.
"""
function honeycomb_unit_cell(site::Int, Ny::Int)::Tuple{Int,Int}
    unit_cell_index = div(site - 1, 2)
    i = div(unit_cell_index, Ny) + 1
    j = mod(unit_cell_index, Ny) + 1
    return (i, j)
end
#Original calculation:


let
    Nx = 5
    Ny = 5
    N_basis = 1  # Number of basis states per site (S=1/2)
    N = Nx * Ny*N_basis
    yperiodic = true
    # Initialize the site degrees of freedom.
    sites = siteinds("S=1/2", N; conserve_qns=true)

    # Use the AutoMPO feature to create the 
    # next-neighbor Heisenberg model.
    ops = OpSum()
    lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
    print(lattice)
    num_bonds = length(lattice)
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
    nsweeps = 5
    maxdim = [10, 20, 100, 100, 200]
    cutoff = [1E-8]
    noise = 1E-7,1E-8,0.0
    # Begin the DMRG calculation
    energy, psi = dmrg(H, psi0;nsweeps, maxdim, cutoff, noise)

    # Print the final energy reported by DMRG
    println("\nGround State Energy = ", energy)
    println("Energy per site = ", energy / N)
    println("\nUsing overlap = ", inner(psi, Apply(H, psi)))

    println("\nTotal QN of Ground State = ", totalqn(psi))

    return
end