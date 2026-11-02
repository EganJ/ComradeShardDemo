# Sharded Comrade posterior with Reactant

`main.jl` builds a VLBI movie posterior, places it on every Reactant device with Comrade's
`todevice(post, ShardLayout(mesh; Ti = :d))`, compiles the log density and its Enzyme
reverse-mode gradient, checks both against the plain CPU path, and times them against one
device and the CPU.

The model:

- EHT 2017 M87 coverage (`data/SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits`), either
  scan-averaged (274 visibilities) or unaveraged (8645).
- The sky is a movie of `NT` frames of `NPIX × NPIX` pixels on a 150 μas field. Each frame
  has its own GMRF log-image (`corr_image_prior`, with its own correlation length) around a
  crescent whose position angle rotates over the observation. The frames are normalized to
  0.6 Jy and drawn with a 3rd-order B-spline pulse.
- The instrument is a complex gain per station and scan (`IIDSitePrior(ScanSeg(), …)`), with
  the phase referenced by `SensitiveReference`.
- The data are simulated from a prior draw with thermal noise.
- The log density and gradient are evaluated at a different prior draw.

Sharding is over the movie frames (`Ti`). Each device holds `NT / ndevices` frames and the
visibilities that fall in them, padded to a common length. The parameter vector is
replicated.

## Setup

The repository is self-contained. The unreleased packages are vendored under `pkgs/` and the
manifest refers to them by relative path; Reactant comes from GitHub `main` at the commit
recorded in `Manifest.toml`. The manifest was resolved with Julia 1.13.3; use the same minor
version.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running

```sh
./run.sh                                   # 4 forced host CPU devices, NPIX=32, NT=8, scan-averaged
NDEV=8 NT=16 ./run.sh                      # 8 forced host CPU devices
AVG=none NPIX=128 NT=16 FRAMES=count NOPT=20 ./run.sh
GPU=1 NT=16 ./run.sh                       # every visible GPU (multi-GPU node)
```

`run.sh` sets the following, then runs `julia -t $JT --project=. main.jl` under a timeout:

- `JULIA_LOAD_PATH`
- `XLA_FLAGS=--xla_force_host_platform_device_count=$NDEV` and `CUDA_VISIBLE_DEVICES=""` for
  CPU runs
- `DEMO_BACKEND`

Reactant must use the IFRT runtime; `LocalPreferences.toml` sets `xla_runtime = "IFRT"`.
The script uses every device in `Reactant.devices()`, and `NT` must be a multiple of the
device count.

| variable | meaning | default |
|---|---|---|
| `NPIX` | pixels per side of each frame | 32 |
| `NT` | number of frames | 2 × devices |
| `AVG` | `scan` (scan-averaged) or `none` | `scan` |
| `FRAMES` | `time`: frames of equal duration; `count`: equal numbers of visibilities | `time` |
| `MODES` | which of `cpu,single,shard` to run | all |
| `NEVAL` | timed calls per measurement | 20 |
| `HLO` | `1`: print the collectives of the partitioned XLA HLO, write `out_*.hlo` | 1 |
| `NOPT` | Adam steps on the sharded posterior | 0 |
| `JT`, `TMO` | Julia threads, timeout in seconds | 8, 3600 |

The script always computes the CPU log density and gradient at the same point as the
reference for the correctness check.

## Measured here

Machine: 20-core laptop, Julia 1.13.3, `-t 8`, 4 devices forced on the host CPU, Reactant
0.2.304 (main). Times are medians of 10 calls (3 for the smallest case). Each call includes
copying the scalar or gradient back to the host. "Compile" is the time for
`@compile sync=true`, which includes tracing. The CPU column is plain Julia: NFFT on the
host, Enzyme on the CPU. Its first gradient call, which includes Enzyme compilation, takes
57–83 s.

| case | path | compile ld / grad | log density | gradient |
|---|---|---|---|---|
| NPIX 16, NT 4, scan avg (274 vis, 1256 params) | CPU | – | 0.18 ms | 0.46 ms |
| | Reactant, 1 device | 45 s / 11 s | 0.18 ms | 0.67 ms |
| | Reactant, 4 devices | 13 s / 7.5 s | 10.0 ms | 8.7 ms |
| NPIX 32, NT 8, scan avg (defaults, 274 vis, 8428 params) | CPU | – | 0.44 ms | 1.46 ms |
| | Reactant, 1 device | 50 s / 15 s | 0.54 ms | 1.21 ms |
| | Reactant, 4 devices | 14 s / 11 s | 13.6 ms | 13.5 ms |
| NPIX 32, NT 8, unaveraged (8645 vis, 8428 params) | CPU | – | 0.94 ms | 3.0 ms |
| | Reactant, 1 device | 47 s / 13 s | 2.2 ms | 9.4 ms |
| | Reactant, 4 devices | 15 s / 11 s | 14.4 ms | 18.1 ms |
| NPIX 64, NT 16, unaveraged (65780 params) | CPU | – | 3.8 ms | 11.1 ms |
| | Reactant, 1 device | 54 s / 19 s | 3.3 ms | 11.5 ms |
| | Reactant, 4 devices | 15 s / 16 s | 14.4 ms | 22.3 ms |
| NPIX 128, NT 16, unaveraged, `FRAMES=count` (262388 params) | CPU | – | 7.7 ms | 33.9 ms |
| | Reactant, 1 device | 56 s / 27 s | 11.2 ms | 27.2 ms |
| | Reactant, 4 devices | 19 s / 24 s | 19.3 ms | 35.7 ms |

The first `@compile` in a process also pays for compiling Reactant's tracing code, so the
first Reactant row of each run has the larger log-density compile time.

**Correctness.** In every case the sharded and single-device values agree with the CPU path:

- log density: relative error 1e-11 to 8e-11;
- gradient: relative 2-norm error 2e-10 to 5e-10.

The sharded and single-device results agree with each other to the last few digits.

**Optimization.** 20 Adam steps compiled as one sharded step function take:

- 10.9 ms per step at NPIX 16 / NT 4;
- 15.6–17.0 ms per step at NPIX 32 / NT 8;
- 35 ms per step at NPIX 128 / NT 16.

The log density increases over those steps, for example from −1.6e7 to −7.5e6 at NPIX 128.

**Memory.** The peak RSS of the whole process is 2.2 GiB at NPIX 16 and 3.1 GiB at NPIX 128.
That covers the CPU, single-device and sharded posteriors, and their compiled programs. The
devices are host memory, so device buffers are included in the RSS.

## Is the sharding real?

Yes, by three checks:

- **Device arrays.** Every device array in the placed posterior has a sharding. The data,
  weights and per-visibility gain indices are split by rows, for example rows 1:100, 101:200,
  … of 400 slots at NPIX 16.
- **Partitioned XLA HLO** (`@code_xla`, written to `out_logdensity.hlo` and
  `out_gradient.hlo`):
  - Each device's FFT covers only its own frames: `c128[1,32,32]` for 4 frames,
    `c128[2,64,64]` for 8 frames on 4 devices.
  - All per-visibility arrays have the per-device length, for example `[2172, …]` of 8688
    slots. None has the full length.
- **Collectives in the partitioned program.**
  - The log density has 1–2 all-reduces, the sums of the log likelihood and the prior. At
    NPIX 32 / NT 8 it also has 3 small collective-permutes of single frames.
  - The gradient has 1–2 all-reduces, including one fused all-reduce of the gradient with
    respect to the replicated parameters, plus 3–6 small collective-permutes of single frames.
  - There are no all-gathers and no `while` loops.

## Why there is no speedup here

The 4 sharded devices are slower than 1 device at every size tried. Their extra time is
roughly constant, about 8–12 ms per call, while the single-device time grows with the
problem. The causes:

1. **Collective latency on simulated CPU devices.** A micro-benchmark in the same
   environment, the sum of a 400-element array split over 4 forced host devices, takes
   7–8 ms per call against 19 µs on one device. Elementwise work on the same split array
   costs 29 µs. The cost is the same with 1 or 8 Julia threads and with 2 or 4 devices, so it
   is the latency of XLA's CPU collectives, not data volume. Each all-reduce or
   collective-permute in the posterior pays it. On GPUs with NCCL the same collectives
   typically take tens of microseconds; this was not measured here.
2. **The devices share the same cores.** One XLA CPU device already runs its FFTs and fusions
   on several cores. Four simulated devices split the same 20 cores, so the compute that
   sharding removes from each device is not freed hardware.
3. **The problems that fit in memory here are small.** One device needs only 3–27 ms per
   gradient, which is comparable to the collective latency alone.

So these numbers show that the sharded program is correct and genuinely partitioned. They do
not show a speedup, and none should be inferred from them. A speedup needs real devices
(multi-GPU) and enough per-device work. Scale up with `NPIX`, `NT` and `AVG=none`, or use
more data.

## Observations for scaling up

- **Frame edges set the padding.** With `FRAMES=time` (equal durations) the 8645 unaveraged
  visibilities fill 12920 slots, because the device with the most points sets the per-device
  length: 49% padding. `FRAMES=count` picks frame edges with equal numbers of visibilities
  and gets 8688 slots (0.5% padding).
- **The prior and the frame construction are not split.** The parameters are replicated, so
  every device evaluates the GMRF prior and the transform of all frames. The partitioner then
  moves some single frames between devices with collective-permutes. Their source is the
  `cat` of per-frame fluctuations in `movie_sky`. This replicated work does not shrink with
  more devices.
- **Single GPU.** With `GPU=1` on the local RTX 5060 Laptop GPU (compute capability 12.0),
  compilation fails in XLA autotuning with `hipGetFuncBySymbol: hipError_t(98)`. Error 98
  appears to be "invalid device function". With `GPU_XLA_FLAGS=--xla_gpu_autotune_level=0`
  both programs compile, but the first call fails with
  `INTERNAL: Unable to launch fft with type IFFT`. The likely cause is that the CUDA/XLA
  build bundled with this Reactant has no kernels for this GPU's architecture. This was not
  investigated further, and a multi-GPU run was not possible here.

## Files

- `main.jl`: the demo.
- `run.sh`: environment and launch.
- `Project.toml`, `Manifest.toml`: the environment, with the packages in `pkgs/` by relative
  path and Reactant from GitHub `main`.
- `pkgs/`: source snapshots (`src`, `ext`, `Project.toml`, `LICENSE`, `README.md`) of
  Comrade, ComradeBase, VLBISkyModels, VLBIInstrumentModels, VLBIImagePriors,
  VLBILikelihoods, VLBIData, InterferometricModels, FieldDimArrays and ReactantCore, as run
  for the numbers above.
- `LocalPreferences.toml`: the IFRT runtime preference.
- `logs/`: the runs the table comes from (`final_default.txt` is the default configuration).
- `out_*.hlo`: the partitioned HLO written by a run with `HLO=1` (not tracked).
