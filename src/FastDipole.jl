module FastDipole

using LinearAlgebra
using StaticArrays
using OffsetArrays
using TOML
using SpecialFunctions
using Parameters
using FFTW
using Tullio
using CrystalInfoFramework
using FilePaths
using ProgressMeter
import Random

include("Util.jl")

include("Lattice.jl")
export Lattice, ReciprocalLattice
export volume, eachcellindex, gen_reciprocal

include("Symmetry.jl")
import .Symmetry
import .Symmetry: Crystal, Bond, print_bond_table, subcrystal, allowed_J
import .Symmetry: all_symmetry_related_bonds
export Crystal, Bond, print_bond_table, subcrystal, allowed_J
export all_symmetry_related_bonds

include("Interactions.jl")
export ExternalField, Heisenberg, OnSite, DiagonalCoupling, GeneralCoupling
export gen_interaction, Hamiltonian
export DipoleReal, DipoleFourier

include("Systems.jl")
export ChargeSystem, SpinSystem, rand!, energy

include("Ewald.jl")
export ewald_sum_monopole, ewald_sum_dipole
export precompute_monopole_ewald, precompute_dipole_ewald
export contract_monopole, contract_dipole
export precompute_monopole_ewald_compressed, precompute_dipole_ewald_compressed
export contract_monopole_compressed, contract_dipole_compressed

include("FourierAccel.jl")

include("Integrators.jl")
export HeunP, HeunP2, LangevinHeunP, evolve!

include("Parsing.jl")
export parse_config

include("StructureFactors.jl")
export diag_structure_factor, full_structure_factor, dipole_form_factor

include("Plotting.jl")
export plot_lattice, plot_spins, plot_bonds, plot_all_bonds, anim_integration, live_integration

end