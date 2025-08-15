# ITensors_integration

This directory contains code and utilities for integrating [ITensors.jl](https://itensor.org/) with [Sunny.jl](https://github.com/SunnySuite/Sunny.jl), primarily for DMRG related methodology but with extensions to TDVP, TEBD for certain systems. See [lattice_helper.jl] for simple system set ups that can be used to test the DMRG integration, though this should work for any 2D system of your choosing. Time evolution using methods such as these should be possible for 1D systems, assuming the fourier transform in useful_TDVP_functions.jl [compute_S()] is set up correctly. 

If you peer inside of dev_testing/. you will find attempts at time evolution for 2D systems which are yet to be successful. I have set up the code such that the quantum correlations computed in a function like [compute_G()] is mapped back to its original 2D structure (see [map_G_to_2D()]), and that data in the QuantumCorrelations object can take 2D momenta values. This should mean it is really only a question of correcting the 2D fourier transform to reflect the more complicated site and phase structure of these 2D systems (See [compute_S_2D()] for an attempt at this).

A note on naming convention: files that follow the CamelCase naming style replicate files that already exist within Sunny but applied to the quantum case. Files that are in snake_case are files that aid the integration between Sunny and ITensor, written by me.

This code was written by me, Adam Lavender, as part of a 2025 summer internship, under the support of Harry Lane to whom I am very grateful. It is very likely that there will be significant bugs and errors in this code but I hope it is a good jumping off point for a useful tool. If you have any questions or concerns, I will do my best to help, please email adam.lavender@student.manchester.ac.uk.

- Adam Lavender, August 2025

## Directory Structure

.
├── dev_testing
│   ├── 2D_FT_testing.jl
│   ├── 2D_multiatom_FT_testing.jl
│   ├── chain_DSSF_TDVP.jl
│   ├── chain_DSSF_TEBD.jl
│   ├── CuGeO_3_DMRG.jl
│   ├── CuGeO_3_LSWT.jl
│   ├── deprecated_plotting.jl
│   ├── itensor_calc.jl
│   └── square_lattice_TDVP.jl
├── examples
│   ├── AFM_chain_TDVP.jl
│   └── kagome_DMRG.jl
├── .DS_Store
├── CorrelationMeasuring.jl
├── ITensors_integration.jl
├── lattice_helpers.jl
├── MeasuredCorrelations.jl
├── overloaded_intensities.jl
├── README.md
├── sunny_toITensor.jl
└── useful_TDVP_functions.jl

3 directories, 20 files

- **examples/**  
  Example scripts demonstrating how to use the integration utilities for specific systems (e.g., AFM chains, material-specific models).

- **useful_TDVP_functions.jl**  
  Core functions and parameter structures for running TDVP time evolution, computing correlation functions, and performing Fourier transforms to obtain dynamical structure factors.

- **ITensors_integration.jl**  
  Binds all the code together into a module like structure that can then be imported as a single file. 
 

## Key Features
- **Ground State Search:**  
  DMRG routines for finding ground states of 1D and quasi-1D quantum spin systems.

- **Time Evolution:**  
  TDVP-based time evolution for computing real-time dynamics and correlation functions.

- **Correlation and Structure Factor Calculation:**  
  Utilities for computing time-dependent correlation functions and their Fourier transforms to obtain \( S(q, \omega))

- **Example Workflows:**  
  Scripts in the `examples/` subdirectory show how to set up and run calculations for various models.

## Getting Started

1. **Install dependencies:**  
   Make sure you have [ITensors.jl](https://github.com/ITensor/ITensors.jl), [Sunny.jl](https://github.com/SunnySuite/Sunny.jl), and [GLMakie.jl](https://github.com/MakieOrg/Makie.jl) installed.

2. **Explore example scripts:**  
   See the `examples/` directory for ready-to-run scripts demonstrating ground state search, time evolution, and structure factor calculations.



