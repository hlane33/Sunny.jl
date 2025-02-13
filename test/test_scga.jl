@testitem "diamond_lattice" begin
    # test against JuliaSCGA (S. Gao)
    tol = 1e-8
    dia = [0.7046602277469309, 0.8230846832863896, 0.23309034250417973, 0.40975668535137943, 0.8474163786642979, 0.8230846832694241, 0.723491683211756, 0.5939752161027589, 0.6506966347286152, 0.8012263819500781, 0.23309034265153963, 0.5939752161512792, 0.7779185770415442, 0.9619923476121188, 0.28363795492206234, 0.4097566857133061, 0.6506966347914551, 0.9619923474164132, 0.8576708385848646, 0.45534457001475764, 0.8474163779976567, 0.8012263817424181, 0.2836379554342603, 0.4553445702621035, 0.9352400940660723]
    a = 8.5031 # (Å)
    latvecs = lattice_vectors(a, a, a, 90, 90, 90)
    cryst = Crystal(latvecs, [[0,0,0]], 227, setting="1")
    dims = (1, 1, 1)
    spininfos = [1 => Moment(; s=1, g=1)]
    sys = System(cryst, spininfos, :dipole;dims, seed=0)
    set_exchange!(sys, -1, Bond(1, 3, [0,0,0]))
    set_exchange!(sys, 0.25, Bond(1, 2, [0,0,0]))
    measure = ssf_perp(sys;)
    scga = Sunny.SCGA(sys;measure)
    kT = 15*meV_per_K
    S=1
    γ=S*(S+1)*length(cryst.positions)
    grid = q_space_grid(cryst, [1, 0, 0], 0:0.9:4, [0, 1, 0], 0:0.9:4; orthogonalize=true)
    res = Sunny.intensities_static(scga, grid; kT, SumRule="Classical",sublattice_resolved=false)
    # Gao had 2 basis sites
    @test abs(sum(reshape(res.data,size(dia))/γ -dia/2))/length(grid.qs)< tol
end

@testitem "square_lattice" begin
    # test against JuliaSCGA (S. Gao)
    tol = 1e-5
    sq=[0.3578208506624103, 0.3850668587212228, 0.4211633125595451, 0.3930558742921638, 0.3586715762816389, 0.3775147329089877, 0.4188431376589759, 0.4009835864744404, 0.36119385315852837, 0.3850668587212228, 0.4063051066815142, 0.4182331099724885, 
    0.36751592492720886, 0.32829927921801794, 0.3490454303292574, 0.4062747596231383, 0.41626772462487194, 0.38789161791308147, 0.4211633125595451, 0.4182331099724885, 0.3710074695764304, 0.2924827325221696, 0.253584799014828, 0.2732638980307635, 
    0.34406882609120737, 0.409719043169102, 0.4213848627804445, 0.3930558742921638, 0.36751592492720886, 0.2924827325221696, 0.21946942640699554, 0.18901809131848027, 0.2041521166894061, 0.26469888441402106, 0.3466451094806684, 0.3903985131607334, 
    0.3586715762816389, 0.32829927921801794, 0.25358479901482806, 0.18901809131848027, 0.16311271576100106, 0.17594081183630647, 0.2284557904965832, 0.3060310966855855, 0.35526653426084637, 0.3775147329089877, 0.3490454303292574, 0.2732638980307635, 
    0.20415211668940608, 0.17594081183630647, 0.18993269166531307, 0.2466353934125701, 0.32714470142342356, 0.37442377142308597, 0.4188431376589759, 0.4062747596231383, 0.34406882609120737, 0.26469888441402106, 0.2284557904965832, 0.2466353934125701, 
    0.3154328042516787, 0.3918720231667212, 0.4179005987958521, 0.4009835864744404, 0.41626772462487194, 0.409719043169102, 0.3466451094806684, 0.3060310966855855, 0.32714470142342356, 0.3918720231667212, 0.42107414035570556, 0.40321330759757756, 
    0.36119385315852837, 0.38789161791308147, 0.4213848627804445, 0.3903985131607334, 0.35526653426084637, 0.37442377142308597, 0.4179005987958521, 0.40321330759757756, 0.3645203324273335];
    a = 1 # (Å)
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 90)
    cryst = Crystal(latvecs, [[0,0,0]])
    dims = (1, 1, 1)
    spininfos = [1 => Moment(; s=1, g=1)]
    sys = System(cryst, spininfos, :dipole; seed=0,dims)
    set_exchange!(sys, -1, Bond(1, 1, [1,0,0]))
    set_exchange!(sys, 0.5, Bond(1, 1, [1,1,0]))
    set_exchange!(sys, 0.25, Bond(1, 1, [2,0,0]))
    measure = ssf_perp(sys)
    scga = Sunny.SCGA(sys;measure, regularization=1e-8)
    kT = 27.5*meV_per_K
    S=1
    γ=S*(S+1)
    grid = q_space_grid(cryst, [1, 0, 0], -1:0.12:0, [0, 1, 0], -1:0.12:0; orthogonalize=true)
    res = Sunny.intensities_static(scga, grid; kT, SumRule="Classical")
    @test abs(sum(reshape(res.data,size(sq))/γ-sq))/length(grid.qs) <tol
end


@testitem "MgCr2O4" begin
    # test against Conlon and Chalker
    tol = 1e-5
    MgCrO=[7.9455280e-02   1.2327507e-01   1.5232493e-01   1.5681329e-01   1.3522730e-01   9.4842745e-02   4.9271681e-02   1.3874324e-02   5.8173439e-04   1.3874324e-02   4.9271681e-02   9.4842745e-02   1.3522730e-01   1.5681329e-01   1.5232493e-01   1.2327507e-01   7.9455280e-02   1.2327507e-01   1.2948879e-01   8.4183394e-02   7.7229764e-02   1.1208030e-01   1.4030772e-01   7.0316789e-02   1.9533030e-02   2.6819210e-03   1.9533030e-02   7.0316789e-02   1.4030772e-01   1.1208030e-01   7.7229764e-02   8.4183394e-02   1.2948879e-01   1.2327507e-01   1.5232493e-01   8.4183394e-02   3.7351609e-02   3.2827340e-02   6.0965125e-02   1.4446159e-01   9.6520123e-02   3.1267583e-02   1.5915070e-02   3.1267583e-02   9.6520123e-02   1.4446159e-01   6.0965125e-02   3.2827340e-02   3.7351609e-02   8.4183394e-02   1.5232493e-01   1.5681329e-01   7.7229764e-02   3.2827340e-02   2.8736540e-02   5.4727377e-02   1.4268618e-01   1.0416464e-01   3.6779356e-02   2.3016168e-02   3.6779356e-02   1.0416464e-01   1.4268618e-01   5.4727377e-02   2.8736540e-02   3.2827340e-02   7.7229764e-02   1.5681329e-01   1.3522730e-01   1.1208030e-01   6.0965125e-02   5.4727377e-02   8.9556374e-02   1.4532790e-01   7.7967759e-02   2.1737224e-02   5.0274892e-03   2.1737224e-02   7.7967759e-02   1.4532790e-01   8.9556374e-02   5.4727377e-02   6.0965125e-02   1.1208030e-01   1.3522730e-01   9.4842745e-02   1.4030772e-01   1.4446159e-01   1.4268618e-01   1.4532790e-01   1.1381744e-01   5.7146070e-02   1.6820885e-02   8.8841047e-04   1.6820885e-02   5.7146070e-02   1.1381744e-01   1.4532790e-01   1.4268618e-01   1.4446159e-01   1.4030772e-01   9.4842745e-02   4.9271681e-02   7.0316789e-02   9.6520123e-02   1.0416464e-01   7.7967759e-02   5.7146070e-02   2.8819752e-02   5.6552242e-03   3.4850119e-04   5.6552242e-03   2.8819752e-02   5.7146070e-02   7.7967759e-02   1.0416464e-01   9.6520123e-02   7.0316789e-02   4.9271681e-02   1.3874324e-02   1.9533030e-02   3.1267583e-02   3.6779356e-02   2.1737224e-02   1.6820885e-02   5.6552242e-03   8.6411134e-04   4.9359139e-04   8.6411134e-04   5.6552242e-03   1.6820885e-02   2.1737224e-02   3.6779356e-02   3.1267583e-02   1.9533030e-02   1.3874324e-02   5.8173439e-04   2.6819210e-03   1.5915070e-02   2.3016168e-02   5.0274892e-03   8.8841047e-04   3.4850119e-04   4.9359139e-04   3.2982024e-14   4.9359139e-04   3.4850119e-04   8.8841047e-04   5.0274892e-03   2.3016168e-02   1.5915070e-02   2.6819210e-03   5.8173439e-04   1.3874324e-02   1.9533030e-02   3.1267583e-02   3.6779356e-02   2.1737224e-02   1.6820885e-02   5.6552242e-03   8.6411134e-04   4.9359139e-04   8.6411134e-04   5.6552242e-03   1.6820885e-02   2.1737224e-02   3.6779356e-02   3.1267583e-02   1.9533030e-02   1.3874324e-02   4.9271681e-02   7.0316789e-02   9.6520123e-02   1.0416464e-01   7.7967759e-02   5.7146070e-02   2.8819752e-02   5.6552242e-03   3.4850119e-04   5.6552242e-03   2.8819752e-02   5.7146070e-02   7.7967759e-02   1.0416464e-01   9.6520123e-02   7.0316789e-02   4.9271681e-02   9.4842745e-02   1.4030772e-01   1.4446159e-01   1.4268618e-01   1.4532790e-01   1.1381744e-01   5.7146070e-02   1.6820885e-02   8.8841047e-04   1.6820885e-02   5.7146070e-02   1.1381744e-01   1.4532790e-01   1.4268618e-01   1.4446159e-01   1.4030772e-01   9.4842745e-02   1.3522730e-01   1.1208030e-01   6.0965125e-02   5.4727377e-02   8.9556374e-02   1.4532790e-01   7.7967759e-02   2.1737224e-02   5.0274892e-03   2.1737224e-02   7.7967759e-02   1.4532790e-01   8.9556374e-02   5.4727377e-02   6.0965125e-02   1.1208030e-01   1.3522730e-01   1.5681329e-01   7.7229764e-02   3.2827340e-02   2.8736540e-02   5.4727377e-02   1.4268618e-01   1.0416464e-01   3.6779356e-02   2.3016168e-02   3.6779356e-02   1.0416464e-01   1.4268618e-01   5.4727377e-02   2.8736540e-02   3.2827340e-02   7.7229764e-02   1.5681329e-01   1.5232493e-01   8.4183394e-02   3.7351609e-02   3.2827340e-02   6.0965125e-02   1.4446159e-01   9.6520123e-02   3.1267583e-02   1.5915070e-02   3.1267583e-02   9.6520123e-02   1.4446159e-01   6.0965125e-02   3.2827340e-02   3.7351609e-02   8.4183394e-02   1.5232493e-01   1.2327507e-01   1.2948879e-01   8.4183394e-02   7.7229764e-02   1.1208030e-01   1.4030772e-01   7.0316789e-02   1.9533030e-02   2.6819210e-03   1.9533030e-02   7.0316789e-02   1.4030772e-01   1.1208030e-01   7.7229764e-02   8.4183394e-02   1.2948879e-01   1.2327507e-01   7.9455280e-02   1.2327507e-01   1.5232493e-01   1.5681329e-01   1.3522730e-01   9.4842745e-02   4.9271681e-02   1.3874324e-02   5.8173439e-04   1.3874324e-02   4.9271681e-02   9.4842745e-02   1.3522730e-01   1.5681329e-01   1.5232493e-01   1.2327507e-01   7.9455280e-02]
    latvecs    = lattice_vectors(8.3342, 8.3342, 8.3342, 90, 90, 90)
    positions  = [[0.1250, 0.1250, 0.1250],
                [0.5000, 0.5000, 0.5000],
                [0.2607, 0.2607, 0.2607]]
    types      = ["Mg","Cr","O"]
    spacegroup = 227 # Space Group Number
    setting    = "2" # Space Group setting
    xtal_mgcro = Crystal(latvecs, positions, spacegroup; types, setting)
    cryst = subcrystal(xtal_mgcro,"Cr")
    dims = (1, 1, 1)  # Supercell dimensions
    spininfos = [1 => Moment(; s=3/2, g=1)]  # Specify spin information, note that all sites are symmetry equivalent
    sys = System(cryst, spininfos, :dipole;dims); # Same on MgCr2O4 crystal
    J1      = 3.27  # value of J1 in meV from Bai's PRL paper
    J_mgcro = [1.00,0.0815,0.1050,0.0085]*J1; # further neighbor pyrochlore relevant for MgCr2O4
    set_exchange!(sys, J_mgcro[1], Bond(1, 2, [0,0,0]))  # J1
    set_exchange!(sys, J_mgcro[2], Bond(1, 7, [0,0,0]))  # J2
    set_exchange!(sys, J_mgcro[3], Bond(1, 3, [1,0,0]))  # J3a -- Careful here!
    set_exchange!(sys, J_mgcro[4], Bond(1, 3, [0,0,0])); # J3b -- And here!
    # values from Conlon + Chalker
    MgCrO=[ 5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   1.8741195e+00   3.2690074e+00   1.8741195e+00   1.3136038e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.2377330e+00   2.8136620e+00   2.4168819e+00   1.8741195e+00   2.4168819e+00   2.8136620e+00   1.2377330e+00   5.5634087e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   6.1505984e-01   1.7143609e+00   2.8136620e+00   3.2690074e+00   2.8136620e+00   1.7143609e+00   6.1505984e-01   1.5971443e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   1.6242284e-01   6.1505984e-01   1.2377330e+00   1.8741195e+00   1.2377330e+00   6.1505984e-01   1.6242284e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02   7.4075451e-02   1.5971443e-01   5.5634087e-01   1.3136038e+00   5.5634087e-01   1.5971443e-01   7.4075451e-02   5.7767400e-02]
    measure = ssf_custom((q, ssf) -> real(sum(ssf)), sys)
    scga = Sunny.SCGA(sys;measure, regularization=1e-8)
    kT = 20*meV_per_K
    grid = q_space_grid(cryst, [1, 0, 0], range(-4, 4, 17), [0, 1, 0], range(-4, 4, 17); orthogonalize=true)
    res = Sunny.intensities_static(scga, grid; kT, SumRule="Quantum",sublattice_resolved = false)
    S = 3/2
    γ=S*(S+1)*length(cryst.positions) 
    @test abs(sum(reshape(res.data,size(MgCrO))/γ - (3/4)*MgCrO)/length(grid.qs))/length(MgCrO) < tol
    # factor 3/4 comes from the fact that C+C solve for a single spin component and 
    # have a four site unit cell.
end

@testitem "Arbitrary Anisotropy" begin
    tol = 1e-7
    # test against JuliaSCGA (S. Gao)
    arb=[0.7586033244771696, 0.7615797236771995, 0.7067376202481463, 0.6713652267703079, 0.7037910523546276, 0.7542251266797739, 0.7483886564702621, 0.7289040202049699, 0.7594392143289115, 0.7113287543280767, 0.6214180238797973, 0.5790787072798553, 0.6192096683970346, 0.7067376202481463, 0.7574706875426616, 0.7617944185477006, 0.704270642625653, 0.6205167495949255, 
    0.5219040226511134, 0.4819720409900168, 0.5213611759220986, 0.6202962239550798, 0.7083751254473052, 0.7350050888005822, 0.6689048865527003, 0.5778569918362061, 0.48158130944975375, 0.4442454028677924, 0.48203511793705517, 0.5800585042005479, 0.6760080974704061, 0.7093415594909234, 0.700944539713289, 0.6175127864675868, 0.5205893697530085, 0.48172879530096047, 
    0.5219040226511135, 0.6218544838522482, 0.7110417061581527, 0.738269833048121, 0.75085044977932, 0.704270642625653, 0.6188902360706803, 0.5792753532751317, 0.6214319191875379, 0.7113287543280766, 0.763936847880277, 0.7690196643410564, 0.7450119108505591, 0.7544416648799901, 0.7062484206003281, 0.6745924555605828, 0.7100415800949798, 0.7633912278503215, 
    0.7586033244771694, 0.7392096662818042, 0.7256714709772182, 0.7586014794304904, 0.7324922896074505, 0.7074913262740532, 0.7368083541302771, 0.768006438327862, 0.7387669446218719, 0.7113287543280766]
    a = 1 # (Å)
    latvecs = lattice_vectors(a, a, 10a, 90, 90, 90)
    cryst = Crystal(latvecs, [[0,0,0]],1)
    dims = (1, 1, 1)
    spininfos = [1 => Moment(; s=1, g=1)]
    sys = System(cryst, spininfos, :dipole; seed=0,dims)
    set_exchange!(sys, -1, Bond(1, 1, [1,0,0]))
    set_exchange!(sys, -1, Bond(1, 1, [0,1,0]))
    set_exchange!(sys, 0.5, Bond(1, 1, [1,1,0]))
    set_exchange!(sys, 0.5, Bond(1, 1, [-1,1,0]))
    set_exchange!(sys, 0.25, Bond(1, 1, [2,0,0]))
    set_exchange!(sys, 0.25, Bond(1, 1, [0,2,0]))
    to_inhomogeneous(sys)
    anis = [0.23 0.56 0.34;
        0.56 0.12 0.45;
        0.34 0.45 0.67]
    anis = 0.5.*(anis + anis')
    set_onsite_coupling!(sys, S -> S'*anis*S, 1)
    k_grid = [0.125:0.125:1;]
    measure = ssf_perp(sys)
    scga = Sunny.SCGA(sys;measure, regularization=1e-8)
    kT = 55*meV_per_K
    grid = q_space_grid(cryst, [1, 0, 0], 0.125:0.125:1, [0, 1, 0], 0.125:0.125:1; orthogonalize=true)
    res = Sunny.intensities_static(scga, grid; kT, SumRule="Classical",sublattice_resolved = false)
    S = 1
    γ=S*(S+1)*length(cryst.positions) 
    @test abs(sum(reshape(res.data,size(arb))-arb))/length(grid.qs) <tol # matches without the S(S+1)Nₐ scaling
end

@testitem "Ferrimagnetic chain" begin 
    tol = 1e-2
    latvecs    = lattice_vectors(3, 5, 8, 90, 90, 90)
    positions  = [[0,0,0],
                [0.5, 0,0]]
    types = ["Ni2","Fe3"]
    cryst = Crystal(latvecs, positions;types)       
    S1 = 1
    S2 = 5/2
    spininfos = [1 => Moment(; s=S1, g=1),2 => Moment(; s=S2, g=1)] 
    sys = System(cryst, spininfos, :dipole;); 
    J1 = 1
    set_exchange!(sys, J1, Bond(1, 2, [0,0,0]))  
    measure = ssf_trace(sys;apply_g = false)
    scga = Sunny.SCGA(sys;measure)
    kT = 22.5*meV_per_K
    qarray = range(0,2,60)
    qs1 = [[qx, 0, 0] for qx in qarray]
    res = Sunny.intensities_static(scga, qs1; kT, SumRule="Classical",sublattice_resolved = true,tol = 1e-6,Nq = 60,λs_init = [7.67033234814451, 1.2272531757030904] )
    sum_rule = S1^2 + S2^2
    @test abs(sum(res.data)/length(qs1)-sum_rule )/sum_rule < tol
end