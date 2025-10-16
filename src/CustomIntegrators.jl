import NonlinearSolve: NonlinearProblem, solve, NewtonRaphson
using Random, LsqFit, StaticArrays

################################################################################
# Colored noise
################################################################################
# T is expected to be either "nothing" or "ColoredNoiseGenerator"
mutable struct ColoredNoiseGenerator

    # Basic integrator parameters
    dt      :: Float64
    kT      :: Float64
    damping :: Float64

    # Temperature dependent noise-generation parameters
    Ω₁ :: Float64
    Ω₂ :: Float64
    Γ₁ :: Float64
    Γ₂ :: Float64
    c₁ :: Float64
    c₂ :: Float64

    # State 
    ζ   :: Array{Float64, 5}
    W1   :: Array{Float64, 5}
    W2   :: Array{Float64, 5}
    u1  :: Array{SVector{2, Float64}, 5}
    u2  :: Array{SVector{2, Float64}, 5}

end

# Solves for when the Planck function reaches 1 percent of its maximum value. Can
# reimplement as simple Newton-Raphson algorithm.
function find_ω_cutoff(kT, α=0.01)
    target = kT*α
    f(ω, p) = ω/(exp(ω/kT)-1) .- p
    prob = NonlinearProblem(f, 5, target)
    sol = solve(prob, NewtonRaphson())
    return max(sol.u, 5.0)
end

planck_spectrum(ω, kT) = iszero(ω) ? kT : ω/(exp(ω/kT) - 1)

function filter_spectrum(ω, p)
    c1, c2, Ω1, Ω2, Γ1, Γ2 = p
    @. ((2c1^2 * Γ1) / ((Ω1^2 - ω^2)^2 + ω^2 * Γ1^2)) + ((2c2^2 * Γ2) / ((Ω2^2 - ω^2)^2 + ω^2 * Γ2^2))
end

@. quad_model(x, p) = p[1] + p[2]*x + p[3]*x*x

function colored_noise_params(kT; α=0.01, N=1000)
    lim = find_ω_cutoff(kT, α)
    ωs = range(0.0, lim, N)
    ys = planck_spectrum.(ωs, kT)

    # Seed optimization correctly -- parameters fit as a function of temperature
    c10 = quad_model(kT,     [0.0, 0.0,      0.3] ) 
    c20 = quad_model(kT,     [0.0, 0.0,      1.88] )
    omega10 = quad_model(kT, [0.0, 1.168055, 0.0])
    omega20 = quad_model(kT, [0.0, 2.748380, 0.0])
    gamma10 = quad_model(kT, [0.0, 3.276618, 0.0])
    gamma20 = quad_model(kT, [0.0, 5.247509, 0.0])

    fit = curve_fit(filter_spectrum, ωs, ys, [c10, c20, omega10, omega20, gamma10, gamma20])
    c₁, c₂, Ω₁, Ω₂, Γ₁, Γ₂ = if all(>(0.0), fit.param)
        fit.param
    else
        error("Didn't find good parameters")
    end

    return (; c₁,  c₂, Ω₁, Ω₂, Γ₁, Γ₂)
end

function ColoredNoiseGenerator(dt; kT, damping, dims, α=0.01, N=1000)
    ## Savin/Barker parameters
    # c₁, c₂ = 1.8315, 0.3429
    # Ω₁, Γ₁ = 2.7189, 5.0142
    # Ω₂, Γ₂ = 1.2223, 3.2974

    c₁, c₂, Ω₁, Ω₂, Γ₁, Γ₂ = colored_noise_params(kT; α, N) 
    ζ = zeros(3, dims...)
    W1 = zeros(3, dims...)
    W2 = zeros(3, dims...)
    u1 = zeros(SVector{2, Float64}, 3, dims...)
    u2 = zeros(SVector{2, Float64}, 3, dims...)

    ColoredNoiseGenerator(
        dt, kT, damping,
        Ω₁, Ω₂, Γ₁, Γ₂, c₁, c₂,
        ζ, W1, W2, u1, u2,
    )
end

function set_temperature!(cng::ColoredNoiseGenerator, kT; α=0.01, N=1000)
    cng.kT = kT

    # Determine new coefficients for noise process
    ## Savin/Barker parameters
    # c₁, c₂ = 1.8315, 0.3429
    # Ω₁, Γ₁ = 2.7189, 5.0142
    # Ω₂, Γ₂ = 1.2223, 3.2974

    c₁, c₂, Ω₁, Ω₂, Γ₁, Γ₂ = colored_noise_params(kT; α, N) 
    cng.c₁ = c₁
    cng.c₂ = c₂
    cng.Ω₁ = Ω₁
    cng.Ω₂ = Ω₂
    cng.Γ₁ = Γ₁
    cng.Γ₂ = Γ₂

    # Reset internal state of noise process
    for i in eachindex(cng.u1)
        cng.u1[i] = zero(SVector{2, Float64})
        cng.u2[i] = zero(SVector{2, Float64})
    end

    return nothing
end

function colored_noise_process_rhs(u, W, dt, Ω, Γ)
    SVector{2, Float64}(
          dt*u[2],
         -dt*(Ω^2*u[1] + Γ*u[2]) + √(2Γ*dt)*W
    )
end

function step!(cng::ColoredNoiseGenerator)
    (; W1, W2, ζ, dt, damping, u1, u2, c₁, c₂, Ω₁, Ω₂, Γ₁, Γ₂) = cng
    randn!(W1)
    randn!(W2)

    for i in eachindex(ζ)
        Δ1 = colored_noise_process_rhs(u1[i], W1[i], dt, Ω₁, Γ₁)
        Δ2 = colored_noise_process_rhs(u1[i] + Δ1, W1[i], dt, Ω₁, Γ₁)
        u1[i] += (Δ1 + Δ2)/2

        Δ1 = colored_noise_process_rhs(u2[i], W2[i], dt, Ω₂, Γ₂)
        Δ2 = colored_noise_process_rhs(u2[i] + Δ1, W2[i], dt, Ω₂, Γ₂)
        u2[i] += (Δ1 + Δ2)/2

        ζ[i] = dt*sqrt(2damping)*(c₁*u1[i][1] + c₂*u2[i][1])
    end
end

function step!(rng, cng::ColoredNoiseGenerator)
    (; W1, W2, ζ, kT, dt, damping, u1, u2, c₁, c₂, Ω₁, Ω₂, Γ₁, Γ₂) = cng
    randn!(rng, W1)
    randn!(rng, W2)

    for i in eachindex(ζ)
        Δ1 = colored_noise_process_rhs(u1[i], W1[i], dt, Ω₁, Γ₁)
        Δ2 = colored_noise_process_rhs(u1[i] + Δ1, W1[i], dt, Ω₁, Γ₁)
        u1[i] += (Δ1 + Δ2)/2

        Δ1 = colored_noise_process_rhs(u2[i], W2[i], dt, Ω₂, Γ₂)
        Δ2 = colored_noise_process_rhs(u2[i] + Δ1, W2[i], dt, Ω₂, Γ₂)
        u2[i] += (Δ1 + Δ2)/2

        ζ[i] = dt*sqrt(2damping)*(c₁*u1[i][1] + c₂*u2[i][1])
    end
end

# function Base.setproperty!(cng::ColoredNoiseGenerator, sym::Symbol, val)
#     if sym == :kT
#         error("Only set temperature of a ColoredNoiseGenerator using `set_temperature!`.")
#     else
#         setfield!(cng, sym, val)
#     end
# end

################################################################################
# Colored Noise Langevin
################################################################################
mutable struct LangevinColored <: Sunny.AbstractIntegrator
    dt              :: Float64
    damping         :: Float64
    kT              :: Float64
    noisesource     :: ColoredNoiseGenerator

    function LangevinColored(dt; λ=nothing, damping=nothing, kT, dims, α=0.01, N=1000)
        if !isnothing(λ)
            @warn "`λ` argument is deprecated! Use `damping` instead."
            damping = @something damping λ
        end
        isnothing(damping) && error("`damping` parameter required")
        iszero(damping) && error("Use ImplicitMidpoint instead for energy-conserving dynamics")

        dt <= 0         && error("Select positive dt")
        kT < 0          && error("Select nonnegative kT")
        damping <= 0    && error("Select positive damping")


        cng = ColoredNoiseGenerator(dt; kT, damping, dims, α, N)
        return new(dt, damping, kT, cng)
    end
end

function LangevinColored(; λ=nothing, damping=nothing, kT, dims)
    Langevin(NaN; λ, damping, kT, dims)
end

function Base.copy(dyn::LangevinColored)
    LangevinColored(dyn.dt; dyn.damping, dyn.kT)
end

function Base.setproperty!(integrator::LangevinColored, sym::Symbol, val)
    if sym == :dt
        setfield!(integrator, sym, val)
        setfield!(integrator.noisesource, sym, val)
    else
        setfield!(integrator, sym, val)
    end
end

function Sunny.step!(sys::System{0}, integrator::LangevinColored)
    (S′, ΔS₁, ΔS₂, ∇E) = Sunny.get_dipole_buffers(sys, 5)
    S = sys.dipoles

    cng = integrator.noisesource
    step!(sys.rng, cng)
    ξ = view(reinterpret(SVector{3, Float64}, cng.ζ), 1, :, :, :, :)

    # Euler prediction step
    Sunny.set_energy_grad_dipoles!(∇E, S, sys)
    Sunny.rhs_dipole!(ΔS₁, S, ξ, ∇E, integrator)
    @. S′ = Sunny.normalize_dipole(S + ΔS₁, sys.κs)

    # Correction step
    Sunny.set_energy_grad_dipoles!(∇E, S′, sys)
    Sunny.rhs_dipole!(ΔS₂, S′, ξ, ∇E, integrator)
    @. S = Sunny.normalize_dipole(S + (ΔS₁+ΔS₂)/2, sys.κs)

    return
end

################################################################################
# Longitudinal Langevin Integaration
################################################################################

# Kluge to coordinate system and CNG: 
mutable struct LongitudinalLangevin{T} <: Sunny.AbstractIntegrator
    dt          :: Float64
    damping     :: Float64
    kT          :: Float64

    A           :: Float64  # Landau Hamiltonian parameter
    B           :: Float64  # Landau Hamiltonian parameter
    C           :: Float64  # Landau Hamiltonian parameter

    noisesource :: T

    function LongitudinalLangevin(dt; λ=nothing, damping=nothing, kT, A, B, C, α=0.01, N=1_000, planck_statistics=nothing)
        if !isnothing(λ)
            @warn "`λ` argument is deprecated! Use `damping` instead."
            damping = @something damping λ
        end
        isnothing(damping) && error("`damping` parameter required")
        iszero(damping) && error("Use ImplicitMidpoint instead for energy-conserving dynamics")

        dt <= 0         && error("Select positive dt")
        kT < 0          && error("Select nonnegative kT")
        damping <= 0    && error("Select positive damping")
        if !isnothing(planck_statistics)
            sys = planck_statistics
            dims = size(sys.dipoles)
            noisesource = ColoredNoiseGenerator(dt; kT, damping, dims, α, N)
            return new{ColoredNoiseGenerator}(dt, damping, kT, A, B, C, noisesource)
        end
        return new{Nothing}(dt, damping, kT, A, B, C, nothing)
    end
end

function Base.setproperty!(integrator::LongitudinalLangevin{ColoredNoiseGenerator}, sym::Symbol, val)
    if sym == :dt
        setfield!(integrator, sym, val)
        setfield!(integrator.noisesource, sym, val)
    else
        setfield!(integrator, sym, val)
    end
end

function LongitudinalLangevin(; λ=nothing, damping=nothing, kT, A, B, C, planck_statistics=nothing)
    LongitudinalLangevin(NaN; λ, damping, kT, A, B, C, planck_statistics)
end

function Base.copy(dyn::LongitudinalLangevin)
    LongitudinalLangevin(dyn.dt; dyn.damping, dyn.kT, dyn.A, dyn.B, dyn.C, dyn.noisesource)
end

# Ma & Dudarev equations
@inline function rhs_dipole_long!(ΔS, S, ξ, ∇E, integrator)
    (; dt, damping) = integrator
    λ = damping

    if iszero(λ)
        @. ΔS = - S × (dt*∇E)
    else
        @. ΔS = - S × (dt*∇E) - dt*λ*∇E + ξ
    end
end

# Heun integration without normalization
function Sunny.step!(sys::Sunny.System{0}, integrator::LongitudinalLangevin{Nothing})
    (S′, ΔS₁, ΔS₂, ξ, ∇E) = Sunny.get_dipole_buffers(sys, 5)
    S = sys.dipoles

    Sunny.fill_noise!(sys.rng, ξ, integrator)

    # Euler prediction step
    Sunny.set_energy_grad_dipoles!(∇E, S, sys)
    set_landau_gradient!(∇E, S, sys, integrator)
    rhs_dipole_long!(ΔS₁, S, ξ, ∇E, integrator)
    @. S′ = S + ΔS₁

    # Correction step
    Sunny.set_energy_grad_dipoles!(∇E, S′, sys)
    set_landau_gradient!(∇E, S′, sys, integrator)
    rhs_dipole_long!(ΔS₂, S′, ξ, ∇E, integrator)
    @. S = S + (ΔS₁+ΔS₂)/2

    return
end


function Sunny.step!(sys::Sunny.System{0}, integrator::LongitudinalLangevin{ColoredNoiseGenerator})
    (S′, ΔS₁, ΔS₂, ξ, ∇E) = Sunny.get_dipole_buffers(sys, 5)
    S = sys.dipoles

    cng = integrator.noisesource
    step!(sys.rng, cng)
    ξ = view(reinterpret(SVector{3, Float64}, cng.ζ), 1, :, :, :, :)

    # Euler prediction step
    Sunny.set_energy_grad_dipoles!(∇E, S, sys)
    set_landau_gradient!(∇E, S, sys, integrator)
    rhs_dipole_long!(ΔS₁, S, ξ, ∇E, integrator)
    @. S′ = S + ΔS₁

    # Correction step
    Sunny.set_energy_grad_dipoles!(∇E, S′, sys)
    set_landau_gradient!(∇E, S′, sys, integrator)
    rhs_dipole_long!(ΔS₂, S′, ξ, ∇E, integrator)
    @. S = S + (ΔS₁+ΔS₂)/2

    return
end

function set_landau_gradient!(∇E, dipoles, sys::Sunny.System{0}, integrator::LongitudinalLangevin)
    (; A, B, C) = integrator
    for site in Sunny.eachsite(sys)
        S = dipoles[site]
        SS = S⋅S
        ∇E[site] += S*(2A + 4B*SS + 6C*SS^2)
    end
end

function Sunny.energy(sys::Sunny.System{0}, integrator::LongitudinalLangevin{T}) where T
    (; A, B, C) = integrator
    E = Sunny.energy(sys; check_normalization=false)
    for site in eachsite(sys)
        S = sys.dipoles[site]
        SS = S⋅S
        E += A*SS + B*SS^2 + C*SS^3
    end
    return E
end

function Sunny.energy_per_site(sys::Sunny.System{0}, integrator::LongitudinalLangevin{T}) where T
    energy(sys, integrator) / length(eachsite(sys))
end