mutable struct ElectronicSystem{N}
    const origin           :: Union{Nothing, System{N}} # System for the original chemical cell

    const crystal          :: Crystal
    const dims             :: NTuple{3, Int}            # Dimensions of lattice in unit cells


    # Interactions may be homogeneous (defined for one unit cell), or
    # inhomogeneous (defined for every cell in the system).
    interactions_union     :: Union{Vector{Interactions}, Array{Interactions, 4}}


    # Dynamical variables and buffers (dims × natoms)
    const extfield         :: Array{Vec3, 4}            # External B field
    const mean_fields          :: Array{Vec2, 4}            # Expected dipoles
    const mean_field_buffers   :: Vector{Array{Vec2, 4}}    # Buffers for dynamics routines

    # Global data
    const rng              :: Random.Xoshiro
end

function ElectronicSystem(crystal::Crystal;
                dims::NTuple{3,Int}=(1, 1, 1), seed=nothing, units=nothing)
    if !isnothing(units)
        @warn "units argument to System is deprecated and will be ignored!"
    end

    # Symops must be non-empty
    validate_symops(crystal)

    # Crystal lattice vectors must be standard (crystal not reshaped)
    @assert isnothing(crystal.root) || crystal.latvecs == crystal.root.latvecs

    na = natoms(crystal)
    # interactions = empty_interactions(mode, na, N)

    mean_fields = fill(zero(Vec2), 1, 1, 1, na)
    mean_field_buffers = Array{Vec2, 4}[]

    rng = isnothing(seed) ? Random.Xoshiro(rand(UInt64, 4)...) : Random.Xoshiro(seed)

    ret = System(nothing, crystal, (1, 1, 1),
                 interactions, extfield, mean_fields, mean_field_buffers, rng)
    return dims == (1, 1, 1) ? ret : repeat_periodically(ret, dims)
end


function dos(E,sys::System,t,U,Vs,Δus,Δds;Γ=0.1,dq= 0.05)
    ρ = 0.0
    qs = Sunny.make_q_grid(sys,dq)  
    Nqs = length(qs)
    for qi in 1:Nqs
        q= qs[qi]
        Eus, Uus = eigen(dynamical_matrix(sys,q,t,U,Vs,Δds))    
        Eds, Uds = eigen(dynamical_matrix(sys,q,t,U,Vs,Δus)) 
        for m in 1:length(Eus)
            ρ += lor(E-Eus[m],Γ)
        end
        for m in 1:length(Eds)
            ρ += lor(E-Eds[m],Γ)
        end  
    end
    ρ /= Nqs
    return ρ
end
