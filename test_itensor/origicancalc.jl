using ITensors, ITensorMPS, Sunny



#Original calculation:


let
    Nx = 8
    Ny = 8
    N = Nx * Ny
    yperiodic = false

    # Initialize the site degrees of freedom.
    sites = siteinds("S=1/2", N; conserve_qns=true)

    # Use the AutoMPO feature to create the 
    # next-neighbor Heisenberg model.
    ops = OpSum()
    lattice = triangular_lattice(Nx, Ny; yperiodic=yperiodic)
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
    println("ORIGINAL CALCULATION ==============================")
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
    println("\nUsing overlap = ", inner(psi, Apply(H, psi)))

    println("\nTotal QN of Ground State = ", totalqn(psi))

    return
end