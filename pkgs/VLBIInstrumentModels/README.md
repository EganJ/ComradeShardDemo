# VLBIInstrumentModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/VLBIInstrumentModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/VLBIInstrumentModels.jl/dev/)
[![Build Status](https://github.com/ptiede/VLBIInstrumentModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/VLBIInstrumentModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/VLBIInstrumentModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/VLBIInstrumentModels.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

VLBI data and instrument models. The package reads, averages and plots visibility tables and
forms closure products from them. It builds instrument models (station gains, leakage, feed
rotation and their priors) and binds them to a table with `observe`. It applies them to model
visibilities with `apply_instrument` and simulates corrupted, noisy tables with
`simulate_observation`. Fitting models to data is done with
[Comrade](https://github.com/ptiede/Comrade.jl), which builds on this package.
