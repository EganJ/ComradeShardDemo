# Sharded gridded-visibility posterior with Reactant

This branch is a simplified version of the demo on `main`. Its program contains only FFTs,
elementwise operations and reductions, with no gathers or scatters:

- the visibilities are gridded onto the cells of a real FFT, so the forward model is an FFT
  of each frame and IF instead of a non-uniform FFT;
- there is no instrument model.

`main.jl` builds the posterior, places it on every Reactant device with its gridded data
split over the frames (`Ti`) and, with `FSHARD > 1`, the IFs (`Fr`), compiles the log density
and its Enzyme reverse-mode gradient, checks both against a plain CPU path, and times them
against one device and the CPU.

The model:

- The coverage is one of (as on `main`):
  - `DATA=eht2017` (default): EHT 2017 M87 coverage
    (`data/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits`), either scan-averaged (274
    visibilities) or unaveraged (8645), one channel.
  - `DATA=synthetic` (`synthetic.jl`): the same stations, source and epoch over `NDAYS` days,
    every baseline of the 2017 data at `TINT`-second integrations in 10-minute scans (2-minute
    gaps) while M87 is at least 10° up at both stations, and `NIF` contiguous 2 GHz IFs of
    `NCHAN` channels each. The defaults give 4.97M visibilities.
  - `DATA=synthetic ARRAY=eht2022`: the same with the 11 EHT 2022 sites in
    `data/array_eht2022.txt`.
- The sky is a movie of `NT` frames of `NPIX × NPIX` pixels on a 150 μas field. Each frame
  has its own GMRF log-image (`corr_image_prior`) around a crescent whose position angle
  rotates over the observation. The frames' priors are one distribution over the
  `NPIX × NPIX × NT` image stack (`FrameGMRFs`), so the parameter vector holds all images,
  then the frames' correlation lengths, and the prior is computed for all frames at once. The frames are normalized to 0.6 Jy. With `DATA=synthetic`
  the image has one plane per IF, scaled by `(ν/ν̄)^α` with a spectral-index parameter `α`.
- Each image plane is zero-padded to `NPAD × NPAD`, `NPAD` the smallest 7-smooth number of
  at least `PADFAC × NPIX`. Its visibilities on the `(NPAD/2 + 1) × NPAD` cells of the
  half-plane real FFT are `response .* conj.(rfft(plane))`, where `response` is the
  visibility of a unit pixel at the grid origin (3rd-order B-spline pulse and phase).
- Each visibility goes to the nearest cell of its frame and IF; one at `u < 0` goes to the
  cell of its conjugate `(-u, -v)`. A cell's weight is the sum of `1/σ²` of its visibilities.
  A visibility outside the grid stops the run.
- The gridded data are simulated from a prior draw with Gaussian noise of variance
  `1/weight` on the cells that hold visibilities. The likelihood is
  `-Σ weight |V - data|² / 2` over all cells.
- The log density and gradient are evaluated at a different prior draw.

The work is set by the grid, `(NPAD/2 + 1) × NPAD × NT × NIF` cells, not by the number of
visibilities: the 4.97M synthetic visibilities fall in 930 of the 67584 cells at the
defaults. Scale the work with `NPIX`, `NT` and `NIF`.

The CPU reference computes the same Fourier sums with dense DFT matrices, split into real
cosine and sine parts, which Enzyme differentiates; it is a separate implementation of the
FFT path. Its cost grows as `NPIX³` per plane.

The data and weights are split along their frame and IF dims on a
`(NDEV / FSHARD) × FSHARD` mesh. The response and the parameter vector are replicated; the
prior and the mean image are constants of the compiled program.

## Setup

The repository is self-contained. The unreleased packages are vendored under `pkgs/` and the
manifest refers to them by relative path; Reactant comes from GitHub `main` at the commit
recorded in `Manifest.toml`. The manifest was resolved with Julia 1.13.3; use the same minor
version.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`./refresh.sh` recopies `pkgs/` from the dev checkouts (`$JULIA_DEV`, default `~/.julia/dev`)
and prints where each copy comes from. The copies are of the working trees:

| package | branch | commit | uncommitted files included |
|---|---|---|---|
| Comrade | `ptiede-redesign` | `fe66b844` | |
| ComradeBase | `ptiede-shard` | `e1f9fc4` | `Project.toml` (version 0.10.0) |
| VLBISkyModels | `ptiede-shard` | `50fd5f4` | |
| VLBIInstrumentModels | `redesign` | `78c4f48` | |
| VLBIImagePriors | `main` | `cfb7b01` | `Project.toml`, `src/markovrf/cache.jl`, `src/markovrf/conditional.jl` |
| VLBILikelihoods | `main` | `5c9f9ca` | `Project.toml` and 7 files in `src` |
| VLBIData | `comradebase-0.10` | `00000000df` | `Project.toml`, `README.md`, `ext/ComradeBaseExt.jl`; `ext/ComradeExt.jl` deleted |
| InterferometricModels | `comradebase-0.10` | `00000000b` | `Project.toml` |
| FieldDimArrays | `reactant-hoist` | `398d08d` | |
| ReactantCore | | | `lib/ReactantCore` of the Reactant commit in `Manifest.toml` |

The timings under "Measured here" were taken with the snapshot above.

## Running

```sh
./run.sh                                   # 4 forced host CPU devices, NPIX=32, NT=8, scan-averaged
DATA=synthetic ./run.sh                    # 4.97M visibilities over 3 days, 4 IFs, 2 × 2 Ti × Fr mesh
DATA=synthetic NPIX=128 NT=32 ./run.sh    # larger grid
GPU=1 NT=16 ./run.sh                       # every visible GPU (multi-GPU node)
BACKEND=tpu NT=16 ./run.sh                 # every TPU core
```

`run.sh` sets the following, then runs `julia -t $JT --project=. main.jl` under a timeout:

- `JULIA_LOAD_PATH`
- `XLA_FLAGS=--xla_force_host_platform_device_count=$NDEV` and `CUDA_VISIBLE_DEVICES=""` for
  `BACKEND=cpu` (the default); `XLA_FLAGS=$ACCEL_XLA_FLAGS` for `gpu`, `tpu` and `trainium`
- `DEMO_BACKEND`, the Reactant backend `main.jl` runs on

Reactant downloads libtpu and the Neuron PJRT plugin when it finds a TPU or a `/dev/neuron*`
device. If the requested backend has no devices, Reactant stops with "No functional client
was found". The demo computes in Float64 and complex128; TPU and Trainium runs have not been
tried, and those devices may not compile 64-bit FFTs.

Reactant must use the IFRT runtime; `LocalPreferences.toml` sets `xla_runtime = "IFRT"`.
The script uses every device in `Reactant.devices()`, and `NT` must be a multiple of the
device count.

| variable | meaning | default |
|---|---|---|
| `DATA` | `eht2017` or `synthetic` | `eht2017` |
| `ARRAY` | stations of the synthetic coverage: `eht2017` or `eht2022` | `eht2017` |
| `NDAYS`, `TINT`, `NIF`, `NCHAN` | synthetic coverage: days, integration (s), IFs, channels per IF | 3, 10, 4, 10 |
| `FSHARD` | devices along `Fr`; `NDEV / FSHARD` split `Ti` | 2 for synthetic data when `NIF` and the device count are even, else 1 |
| `NPIX` | pixels per side of each frame | 32 |
| `PADFAC` | zero-padding factor of the FFT | 2 |
| `NT` | number of frames | 2 × devices |
| `AVG` | `scan` (scan-averaged) or `none` | `scan`; `none` for synthetic data |
| `FRAMES` | `time`: frames of equal duration; `count`: equal numbers of visibilities | `time`; `count` for synthetic data |
| `MODES` | which of `cpu,single,shard` to run | all |
| `NEVAL` | timed calls per measurement | 20 |
| `HLO` | `1`: print the collectives, gathers, scatters and FFTs of the partitioned XLA HLO, write `out_*.hlo` | 1 |
| `NOPT` | Adam steps on the sharded posterior | 0 |
| `JT`, `TMO` | Julia threads, timeout in seconds | 8, 3600 |

The script computes the CPU log density and gradient as the reference for the correctness
check whenever `MODES` includes `cpu`, `single` or `shard`.

## Measured here

Machine: 20-core laptop, Julia 1.13.3, `-t 8`, 4 devices forced on the host CPU, Reactant
0.2.304 (main). Times are medians of 20 calls, each including the copy of the scalar or
gradient back to the host. "Compile" is `@compile sync=true`, including tracing.

| case | path | compile ld / grad | log density | gradient |
|---|---|---|---|---|
| defaults: 274 vis in 73 of 16896 cells, 8201 params | CPU (DFT matrices) | – | 0.32 ms | 2.9 ms |
| | Reactant, 1 device | 28 s / 3.7 s | 0.84 ms | 0.44 ms |
| | Reactant, 4 devices | 2.9 s / 2.2 s | 11.9 ms | 11.6 ms |
| `DATA=synthetic`: 4.97M vis in 930 of 67584 cells, 8202 params, 2 × 2 mesh | CPU (DFT matrices) | – | 0.76 ms | 8.9 ms |
| | Reactant, 1 device | 28 s / 3.9 s | 0.44 ms | 0.66 ms |
| | Reactant, 4 devices | 2.8 s / 1.9 s | 9.7 ms | 10.2 ms |

The runs are in `logs/gridded_default.txt` and `logs/gridded_synthetic.txt`.

**Correctness.** The model visibilities agree with VLBISkyModels' `DFTAlg` on the same cells
to 1e-15, and the FFT path with the DFT matrices to 4e-16. The single-device and sharded log
densities agree with the CPU to 4e-16 or better, the gradients to 7e-15.

**Partitioned HLO** (`@code_xla`, written to `out_logdensity.hlo` and `out_gradient.hlo`):

| case | program | FFT | all-reduce | collective-permute | gather / scatter / all-gather / `while` |
|---|---|---|---|---|---|
| defaults, 4 × 1 mesh | log density | 1 | 1 | 3 | 0 |
| | gradient | 2 | 1 | 6 | 0 |
| synthetic, 2 × 2 mesh | log density | 1 | 2 | 1 | 0 |
| | gradient | 2 | 3 | 2 | 0 |

The all-reduces are the sums of the log likelihood and the gradient with respect to the
replicated parameters. The collective-permutes move single `NPIX × NPIX` frames between
devices; they came from the `cat` of separately stored frames in `movie_image`. These
counts and the timings above were measured with a prior that stored each frame's image and
correlation length separately, before `FrameGMRFs`.

**No speedup here.** As on `main`, 4 simulated CPU devices are slower than 1: each collective
costs a few milliseconds on forced host devices, the devices share the same cores, and the
single-device time at these sizes is under a millisecond. These runs show that the program is
partitioned without gathers or scatters; a speedup needs real devices and a larger grid.

## Files

- `main.jl`: the demo.
- `synthetic.jl`: the synthetic coverage for `DATA=synthetic`.
- `data/array_eht2022.txt`: EHT 2022 station positions and mounts (ehtim array format).
- `run.sh`: environment and launch.
- `refresh.sh`: recopies `pkgs/` from the dev checkouts.
- `Project.toml`, `Manifest.toml`: the environment, with the packages in `pkgs/` by relative
  path and Reactant from GitHub `main`.
- `pkgs/`: source snapshots (`src`, `ext`, `Project.toml`, `LICENSE`, `README.md`) of
  Comrade, ComradeBase, VLBISkyModels, VLBIInstrumentModels, VLBIImagePriors,
  VLBILikelihoods, VLBIData, InterferometricModels, FieldDimArrays and ReactantCore; their
  sources are listed under Setup.
- `LocalPreferences.toml`: the IFRT runtime preference.
- `logs/`: `gridded_*.txt` are the runs of this branch's table; the other logs are runs of
  the model on `main`.
- `out_*.hlo`: the partitioned HLO written by a run with `HLO=1` (not tracked).
