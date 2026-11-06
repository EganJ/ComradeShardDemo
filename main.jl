# Sharded gridded-visibility movie posterior: log density and gradient on all Reactant devices.
#
# The visibilities are gridded onto the cells of the half-plane real FFT of each zero-padded
# frame and IF, so the forward model is an FFT and elementwise products, with no instrument.
#
# Problem size and stages are set by environment variables (see README.md):
#   NPIX   pixels per side of each frame         (default 32)
#   PADFAC zero-padding factor of the FFT        (default 2)
#   NT     number of movie frames                (default 2 × number of devices)
#   DATA   "eht2017" (the coverage in data/) or "synthetic" (see synthetic.jl)  (default "eht2017")
#   ARRAY  stations of the synthetic coverage: "eht2017" or "eht2022" (data/array_eht2022.txt)
#                                                (default "eht2017")
#   NDAYS, TINT, NIF, NCHAN  synthetic coverage: days, integration in s, IFs, channels per IF
#                            (defaults 3, 10, 4, 10)
#   FSHARD devices along Fr; the rest split Ti   (default 2 for synthetic data when it divides, else 1)
#   AVG    "scan" (scan-averaged) or "none"      (default "scan"; "none" for synthetic data)
#   NEVAL  timed evaluations per measurement     (default 20)
#   MODES  comma list of cpu,single,shard        (default "cpu,single,shard")
#   HLO    "1" to inspect the partitioned XLA HLO of the sharded program (default "1")
#   NOPT   Adam steps on the sharded posterior, 0 to skip (default 0)
#   FRAMES "time" (frames of equal duration) or "count" (equal numbers of visibilities)
#          (default "time"; "count" for synthetic data)

const BACKEND = get(ENV, "DEMO_BACKEND", "cpu")
const NPIX = parse(Int, get(ENV, "NPIX", "32"))
const PADFAC = parse(Int, get(ENV, "PADFAC", "2"))
const DATASET = get(ENV, "DATA", "eht2017")
DATASET in ("eht2017", "synthetic") || error("DATA must be eht2017 or synthetic, not $DATASET")
const SYNTH = DATASET == "synthetic"
const ARRAY = get(ENV, "ARRAY", "eht2017")
ARRAY in ("eht2017", "eht2022") || error("ARRAY must be eht2017 or eht2022, not $ARRAY")
SYNTH || ARRAY == "eht2017" || error("ARRAY = $ARRAY needs DATA=synthetic")
const NDAYS = parse(Int, get(ENV, "NDAYS", "3"))
const TINT = parse(Float64, get(ENV, "TINT", "10"))
const NIF = parse(Int, get(ENV, "NIF", "4"))
const NCHAN = parse(Int, get(ENV, "NCHAN", "10"))
const AVG = get(ENV, "AVG", SYNTH ? "none" : "scan")
const NEVAL = parse(Int, get(ENV, "NEVAL", "20"))
const MODES = split(get(ENV, "MODES", "cpu,single,shard"), ",")
const HLO = get(ENV, "HLO", "1") == "1"
const NOPT = parse(Int, get(ENV, "NOPT", "0"))
const FRAMES = get(ENV, "FRAMES", SYNTH ? "count" : "time")

using Reactant
Reactant.set_default_backend(BACKEND)
const DEVS = vec(collect(Reactant.devices()))
const NDEV = length(DEVS)
const NT = parse(Int, get(ENV, "NT", string(2 * NDEV)))
const FSHARD = parse(Int, get(ENV, "FSHARD", string(SYNTH && iseven(NIF) && iseven(NDEV) ? 2 : 1)))
const TSHARD = NDEV ÷ FSHARD

using Comrade, VLBIFiles, VLBIImagePriors, Distributions, Enzyme
using Random, Statistics, LinearAlgebra
using AbstractFFTs: rfft
using ProbabilityTransports: NamedDist, TVFlat, latent_pback, latent_pfwd_and_logdensity, transport_to
import VLBISkyModels

rss() = (s = read("/proc/self/status", String); m(k) = parse(Int, match(Regex("$k:\\s+(\\d+)"), s)[1]) / 2^20; (now = m("VmRSS"), peak = m("VmHWM")))
fmt(t) = t < 1.0e-3 ? string(round(t * 1.0e6; digits = 1), " µs") : t < 1 ? string(round(t * 1.0e3; digits = 2), " ms") : string(round(t; digits = 2), " s")
function stamp(msg)
    r = rss()
    println("[", round(time() - T0; digits = 1), " s | RSS ", round(r.now; digits = 2), " GiB, peak ", round(r.peak; digits = 2), " GiB] ", msg)
    return flush(stdout)
end
const T0 = time()

println("backend = $BACKEND, devices = $NDEV ($(join(unique(string.(typeof.(DEVS))), ", "))), threads = $(Threads.nthreads())")
println("DATA = $DATASET", SYNTH ? " (ARRAY = $ARRAY, NDAYS = $NDAYS, TINT = $TINT s, NIF = $NIF, NCHAN = $NCHAN)" : "", ", mesh = $TSHARD (Ti) × $FSHARD (Fr)")
println("NPIX = $NPIX, PADFAC = $PADFAC, NT = $NT, AVG = $AVG, FRAMES = $FRAMES, NEVAL = $NEVAL, MODES = $(join(MODES, ","))")
NDEV % FSHARD == 0 || error("FSHARD = $FSHARD does not divide the $NDEV devices")
SYNTH || FSHARD == 1 || error("FSHARD = $FSHARD needs DATA=synthetic; the EHT 2017 data have one channel")
NIF % FSHARD == 0 || error("NIF = $NIF IFs do not split evenly over FSHARD = $FSHARD devices")
NT % TSHARD == 0 || error("NT = $NT frames do not split evenly over $TSHARD devices along Ti")

## Data: EHT 2017 M87 coverage, or synthetic coverage of the same stations

const DATA = joinpath(@__DIR__, "data")
uvd = VLBIFiles.load(VLBIFiles.UVData, joinpath(DATA, "SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"))
tab = VisTable(uvd, IPol())
if SYNTH
    include(joinpath(@__DIR__, "synthetic.jl"))
    # 2022 site codes of the 2017 stations, for the 2017 noise.
    stations, aliases = ARRAY == "eht2022" ?
        (read_array(joinpath(DATA, "array_eht2022.txt")), Dict(:AX => :AP, :MG => :AZ, :MM => :JC, :SW => :SM)) :
        (nothing, Dict{Symbol, Symbol}())
    tab, ifedges = synthetic_table(tab; ndays = NDAYS, tint = TINT, nif = NIF, nchan = NCHAN, stations, aliases)
end
tab = AVG == "scan" ? average(tab, ByScan()) : AVG == "none" ? tab : error("AVG must be scan or none")
stamp("loaded $(length(measurement(tab))) visibilities ($(size(domain(tab))[1]) records × $(size(domain(tab))[2]) channels)")

## Sky: a movie, one GMRF-correlated log-image per frame around a rotating crescent

τ = Comrade.times(tab)
edges = if FRAMES == "time"
    range(first(τ) - 1.0e-3, last(τ) + 1.0e-3; length = NT + 1)
elseif FRAMES == "count"
    tv = sort(domain(tab).Ti[:, 1])
    [first(tv) - 1.0e-3; [(tv[k] + tv[k + 1]) / 2 for k in round.(Int, (1:(NT - 1)) .* length(tv) ./ NT)]; last(tv) + 1.0e-3]
else
    error("FRAMES must be time or count")
end
fov = μas2rad(150.0)
g2 = spatialgrid(fov, fov, NPIX, NPIX)
const NPLANE = SYNTH ? NIF : 1

crescent(ξ) = modify(RingTemplate(RadialGaussian(0.15), AzimuthalCosine(0.6, ξ)), Stretch(μas2rad(21.0)))
pas = range(0.0, π / 2; length = NT)
meanimg = stack(ξ -> baseimage(intensitymap(crescent(ξ), g2)), pas)
logmean = log.(meanimg .+ 1.0e-3 * maximum(meanimg))

# Flux per pixel, NPIX × NPIX × frames × IFs.
function movie_image(θ, meta)
    (; c, σimg) = θ
    δ = cat(map(ck -> reshape(ck.params, size(ck.params)..., 1), c)...; dims = 3)
    e = exp.(meta.logmean .+ σimg .* δ)
    img = meta.ftot .* e ./ sum(e; dims = (1, 2))
    return hasproperty(θ, :α) ? img .* reshape(meta.νrel .^ θ.α, 1, 1, 1, :) : reshape(img, size(img)..., 1)
end

cprior = corr_image_prior(g2, tab)
prior = (c = ntuple(_ -> cprior, NT), σimg = VLBIExponential(0.5))
metadata = (; logmean, ftot = 0.6)
if SYNTH
    # One image plane per IF, scaled by (ν / ν̄)^α.
    νif = (ifedges[1] .+ ifedges[2]) ./ 2
    prior = merge(prior, (; α = VLBIGaussian(-0.5, 0.5)))
    metadata = merge(metadata, (; νrel = νif ./ mean(νif)))
end
prior = NamedDist(prior)

## Fourier transform: the half-plane real FFT of each frame and IF, zero-padded to NPAD × NPAD

const NPAD = nextprod((2, 3, 5, 7), PADFAC * NPIX)
const dx, dy = step(g2.X), step(g2.Y)
const Δu, Δv = 1 / (NPAD * dx), 1 / (NPAD * dy)
# Signed frequency index of FFT bin `j` of `n`.
freqindex(j, n) = mod(j - 1 + n ÷ 2, n) - n ÷ 2
const ku = 0:(NPAD ÷ 2)
const kv = freqindex.(1:NPAD, NPAD)

# The Fourier sum Σ img[m, n] cispi(2 (k (m - 1) + l (n - 1)) / NPAD) of each image plane, for
# `k in ku` and `l in kv`, by an FFT of the zero-padded planes.
struct PaddedRFFT
    n::Int
end
function (p::PaddedRFFT)(img)
    pimg = similar(img, (p.n, p.n, size(img)[3:end]...))
    fill!(pimg, 0)
    pimg[axes(img, 1), axes(img, 2), :, :] .= img
    return conj.(rfft(pimg, (1, 2)))
end

# The same sum by dense DFT matrices, `Fx = Cx + i Sx` and `Fy = Cy + i Sy`, kept real for Enzyme.
struct DFTMatrices{M}
    Cx::M
    Sx::M
    Cy::M
    Sy::M
end
function DFTMatrices(n)
    px, py = 2 .* ku .* (0:(n - 1))' ./ NPAD, 2 .* kv .* (0:(n - 1))' ./ NPAD
    return DFTMatrices(cospi.(px), sinpi.(px), cospi.(py), sinpi.(py))
end
function (d::DFTMatrices)(img)
    planes = map(CartesianIndices(axes(img)[3:end])) do I
        A, B = d.Cx * img[:, :, I], d.Sx * img[:, :, I]
        complex.(A * d.Cy' .- B * d.Sy', A * d.Sy' .+ B * d.Cy')
    end
    return stack(planes)
end

# Visibility of a unit-flux pixel at the grid origin in each cell: its pulse and phase.
pulse = BSplinePulse{3}()
response = [
    VLBISkyModels.ω(pulse, k * Δu * dx) * VLBISkyModels.ω(pulse, l * Δv * dy) * cispi(2 * (k * Δu * first(g2.X) + l * Δv * first(g2.Y)))
        for k in ku, l in kv
]

## Gridded data: the inverse-variance weights of the visibilities in each cell, frame and IF

# Weights `Σ 1 / σ²` of the visibilities of `tab` in each half-plane FFT cell, frame and IF;
# a visibility at `(u, v)` with `u < 0` counts at its conjugate `(-u, -v)`.
function gridweights(tab, edges, nplane, ifedges)
    d, σ = domain(tab), Comrade.noise(tab)
    U, V, Ti, Fr = d.U, d.V, d.Ti, d.Fr
    w = zeros(length(ku), length(kv), length(edges) - 1, nplane)
    for I in CartesianIndices(σ)
        i, j = round(Int, U[I] / Δu), round(Int, V[I] / Δv)
        i < 0 && ((i, j) = (-i, -j))
        i <= last(ku) && minimum(kv) <= j <= maximum(kv) ||
            error("a visibility at (u, v) = ($(U[I]), $(V[I])) is outside the FFT grid; increase NPIX")
        k = searchsortedlast(edges, Ti[I[1]])
        f = isnothing(ifedges) ? 1 : searchsortedlast(ifedges[1], Fr[I[2]])
        w[i + 1, mod(j, NPAD) + 1, k, f] += 1 / σ[I]^2
    end
    return w
end
weight = gridweights(tab, edges, NPLANE, SYNTH ? ifedges : nothing)

## Posterior

# Gaussian gridded visibilities `data` with weights `weight` for the model
# `response .* ft(movie_image(θ, meta))`, with the prior of `transform` on flat parameters.
struct GriddedPosterior{T, F, R, D, W, M}
    transform::T
    ft::F
    response::R
    data::D
    weight::W
    meta::M
end

visibilities(post, θ) = post.response .* post.ft(movie_image(θ, post.meta))

function logdensity(post::GriddedPosterior, x)
    θ, ℓ = latent_pfwd_and_logdensity(post.transform, x)
    return ℓ - sum(post.weight .* abs2.(visibilities(post, θ) .- post.data)) / 2
end

rng = Xoshiro(17)
θtrue = merge(rand(rng, prior), (σimg = 0.3,))
ptrans = transport_to(prior, TVFlat())
post_sim = GriddedPosterior(ptrans, DFTMatrices(NPIX), response, nothing, weight, metadata)
noise = (randn(rng, size(weight)) .+ im .* randn(rng, size(weight))) ./ sqrt.(max.(weight, eps()))
data = ifelse.(weight .> 0, visibilities(post_sim, θtrue) .+ noise, zero(ComplexF64))
post_cpu = GriddedPosterior(ptrans, DFTMatrices(NPIX), response, data, weight, metadata)
x0 = latent_pback(ptrans, rand(Xoshiro(41), prior))
const NCELL = count(>(0), weight)
stamp("posterior: $(length(measurement(tab))) visibilities in $NCELL of $(length(weight)) cells ($(size(weight, 1))×$(size(weight, 2)) × $NT frames × $NPLANE IFs), $(NPIX)×$NPIX pixels padded to $NPAD, $(length(x0)) parameters")

fgrad(x, tp) = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Const(Base.Fix1(logdensity, tp)), x)[1]

function bench(f, n)
    f()
    ts = [@elapsed(f()) for _ in 1:n]
    return (; median = median(ts), min = minimum(ts))
end

results = Dict{String, Any}()

## CPU reference (plain Julia, DFT matrices on the host, Enzyme on the CPU)

ld_cpu = gr_cpu = nothing
if "cpu" in MODES || "single" in MODES || "shard" in MODES
    tc = @elapsed ld_cpu = logdensity(post_cpu, x0)
    tg = @elapsed gr_cpu = fgrad(x0, post_cpu)
    stamp("CPU first calls: logdensity $(fmt(tc)), gradient $(fmt(tg)) (includes Julia/Enzyme compilation); logdensity = $ld_cpu")
end
if "cpu" in MODES
    bl = bench(() -> logdensity(post_cpu, x0), NEVAL)
    bg = bench(() -> fgrad(x0, post_cpu), max(3, NEVAL ÷ 4))
    results["cpu"] = (; compile_ld = NaN, compile_grad = NaN, ld = bl, grad = bg)
    stamp("CPU: logdensity $(fmt(bl.median)) (min $(fmt(bl.min))), gradient $(fmt(bg.median)) (min $(fmt(bg.min)))")
end

# Compile, check against the CPU, and time the log density and gradient of `tdev` at `xr`.
function run_reactant(label, tdev, xr)
    tcl = @elapsed ldc = @compile sync = true logdensity(tdev, xr)
    stamp("$label: compiled logdensity in $(fmt(tcl))")
    tcg = @elapsed gc = @compile sync = true fgrad(xr, tdev)
    stamp("$label: compiled gradient in $(fmt(tcg))")
    ld = convert(Float64, ldc(tdev, xr))
    gr = convert(Vector{Float64}, gc(xr, tdev))
    rld = abs(ld - ld_cpu) / abs(ld_cpu)
    rgr = norm(gr - gr_cpu) / norm(gr_cpu)
    stamp("$label: logdensity = $ld, rel. error vs CPU = $rld; gradient rel. error (2-norm) = $rgr, max abs = $(maximum(abs, gr - gr_cpu))")
    bl = bench(() -> convert(Float64, ldc(tdev, xr)), NEVAL)
    bg = bench(() -> convert(Vector{Float64}, gc(xr, tdev)), NEVAL)
    stamp("$label: logdensity $(fmt(bl.median)) (min $(fmt(bl.min))), gradient $(fmt(bg.median)) (min $(fmt(bg.min)))")
    return (; compile_ld = tcl, compile_grad = tcg, ld = bl, grad = bg, rld, rgr), ldc, gc
end

# `post` on the devices with the FFT: data and weights placed by `sharding`, the response by
# `replicated`; the prior and metadata stay on the host, as constants of the compiled program.
place(post::GriddedPosterior, sharding, replicated) = GriddedPosterior(
    post.transform, PaddedRFFT(NPAD), Reactant.to_rarray(post.response; sharding = replicated),
    Reactant.to_rarray(post.data; sharding), Reactant.to_rarray(post.weight; sharding), post.meta,
)

## One device

if "single" in MODES
    tsingle = place(post_cpu, Reactant.Sharding.NoSharding(), Reactant.Sharding.NoSharding())
    results["single"], _, _ = run_reactant("1 device", tsingle, Reactant.to_rarray(x0))
    tsingle = nothing
    GC.gc()
end

## All devices, sharded over the movie frames

# The ops of the HLO text `hlo` that move or index data.
function opcounts(hlo)
    ops = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute", "while", "gather", "scatter", "fft")
    return NamedTuple{Symbol.(replace.(ops, "-" => "_"))}(map(op -> count(Regex("(?<![\\w-])" * op * "(-start)?\\("), hlo), ops))
end

# `n` Adam steps of gradient ascent on `tdev`, compiled as one step function.
function run_adam(tdev, ldc, x0, mesh, n)
    function adam_step(x, m, v, t, tp)
        gx = fgrad(x, tp)
        m = 0.9 .* m .+ 0.1 .* gx
        v = 0.999 .* v .+ 0.001 .* gx .^ 2
        x = x .+ 0.01 .* (m ./ (1 .- 0.9 .^ t)) ./ (sqrt.(v ./ (1 .- 0.999 .^ t)) .+ 1.0e-8)
        return x, m, v, t .+ 1
    end
    rep = Reactant.Sharding.Replicated(mesh)
    x = Reactant.to_rarray(x0; sharding = rep)
    m = Reactant.to_rarray(zero(x0); sharding = rep)
    v = Reactant.to_rarray(zero(x0); sharding = rep)
    t = Reactant.to_rarray([1.0]; sharding = rep)
    tc = @elapsed stepc = @compile sync = true adam_step(x, m, v, t, tdev)
    ld_start = convert(Float64, ldc(tdev, x))
    topt = @elapsed for _ in 1:n
        x, m, v, t = stepc(x, m, v, t, tdev)
    end
    stamp("$n Adam steps on the sharded posterior: compile $(fmt(tc)), run $(fmt(topt)) ($(fmt(topt / n)) per step); logdensity $ld_start -> $(convert(Float64, ldc(tdev, x)))")
    return x
end

if "shard" in MODES && NDEV > 1
    mesh = FSHARD == 1 ? Reactant.Sharding.Mesh(reshape(DEVS, NDEV), (:t,)) :
        Reactant.Sharding.Mesh(reshape(DEVS, TSHARD, FSHARD), (:t, :f))
    datasharding = Reactant.Sharding.NamedSharding(mesh, (nothing, nothing, :t, FSHARD == 1 ? nothing : :f))
    tp = @elapsed tdev = place(post_cpu, datasharding, Reactant.Sharding.Replicated(mesh))
    stamp("sharded posterior placed in $(fmt(tp))")
    for (i, s) in enumerate(tdev.data.sharding.device_to_array_slices)
        println("  device $i holds frames $(s[3]) and IFs $(s[4]) of the gridded data")
    end

    xr = Reactant.to_rarray(x0; sharding = Reactant.Sharding.Replicated(mesh))
    results["shard"], ldc, gc = run_reactant("$NDEV devices", tdev, xr)

    if HLO
        hl = string(Reactant.@code_xla logdensity(tdev, xr))
        hg = string(Reactant.@code_xla fgrad(xr, tdev))
        println("partitioned HLO ops, logdensity: ", opcounts(hl))
        println("partitioned HLO ops, gradient:   ", opcounts(hg))
        write(joinpath(@__DIR__, "out_logdensity.hlo"), hl)
        write(joinpath(@__DIR__, "out_gradient.hlo"), hg)
    end

    NOPT > 0 && run_adam(tdev, ldc, x0, mesh, NOPT)
end
println("\nSUMMARY (median of $NEVAL calls; NPIX = $NPIX, NT = $NT, $NCELL gridded cells, $(length(x0)) parameters)")
for k in ("cpu", "single", "shard")
    haskey(results, k) || continue
    r = results[k]
    name = k == "cpu" ? "CPU (no Reactant)" : k == "single" ? "Reactant, 1 device" : "Reactant, $NDEV devices"
    err = hasproperty(r, :rld) ? ", rel. err ld $(round(r.rld; sigdigits = 2)), grad $(round(r.rgr; sigdigits = 2))" : ""
    println(rpad(name, 22), " compile ld ", lpad(fmt(r.compile_ld), 9), ", grad ", lpad(fmt(r.compile_grad), 9),
        " | logdensity ", lpad(fmt(r.ld.median), 9), " | gradient ", lpad(fmt(r.grad.median), 9), err)
end
stamp("done")
