# Sharded Comrade movie posterior: log density and gradient on all Reactant devices.
#
# Problem size and stages are set by environment variables (see README.md):
#   NPIX   pixels per side of each frame         (default 32)
#   NT     number of movie frames                (default 2 × number of devices)
#   AVG    "scan" (scan-averaged) or "none"      (default "scan")
#   NEVAL  timed evaluations per measurement     (default 20)
#   MODES  comma list of cpu,single,shard        (default "cpu,single,shard")
#   HLO    "1" to inspect the partitioned XLA HLO of the sharded program (default "1")
#   NOPT   Adam steps on the sharded posterior, 0 to skip (default 0)
#   FRAMES "time" (frames of equal duration) or "count" (equal numbers of visibilities)

const BACKEND = get(ENV, "DEMO_BACKEND", "cpu")
const NPIX = parse(Int, get(ENV, "NPIX", "32"))
const AVG = get(ENV, "AVG", "scan")
const NEVAL = parse(Int, get(ENV, "NEVAL", "20"))
const MODES = split(get(ENV, "MODES", "cpu,single,shard"), ",")
const HLO = get(ENV, "HLO", "1") == "1"
const NOPT = parse(Int, get(ENV, "NOPT", "0"))
const FRAMES = get(ENV, "FRAMES", "time")

using Reactant
Reactant.set_default_backend(BACKEND)
const DEVS = vec(collect(Reactant.devices()))
const NDEV = length(DEVS)
const NT = parse(Int, get(ENV, "NT", string(2 * NDEV)))

using Comrade, VLBIFiles, VLBIImagePriors, Distributions, Enzyme
using Random, Statistics, LinearAlgebra

rss() = (s = read("/proc/self/status", String); m(k) = parse(Int, match(Regex("$k:\\s+(\\d+)"), s)[1]) / 2^20; (now = m("VmRSS"), peak = m("VmHWM")))
fmt(t) = t < 1.0e-3 ? string(round(t * 1.0e6; digits = 1), " µs") : t < 1 ? string(round(t * 1.0e3; digits = 2), " ms") : string(round(t; digits = 2), " s")
function stamp(msg)
    r = rss()
    println("[", round(time() - T0; digits = 1), " s | RSS ", round(r.now; digits = 2), " GiB, peak ", round(r.peak; digits = 2), " GiB] ", msg)
    return flush(stdout)
end
const T0 = time()

println("backend = $BACKEND, devices = $NDEV ($(join(unique(string.(typeof.(DEVS))), ", "))), threads = $(Threads.nthreads())")
println("NPIX = $NPIX, NT = $NT, AVG = $AVG, FRAMES = $FRAMES, NEVAL = $NEVAL, MODES = $(join(MODES, ","))")
NT % NDEV == 0 || error("NT = $NT frames do not split evenly over $NDEV devices")

## Data: EHT 2017 M87 coverage

const DATA = joinpath(@__DIR__, "data")
uvd = VLBIFiles.load(VLBIFiles.UVData, joinpath(DATA, "SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"))
tab = VisTable(uvd, IPol())
tab = AVG == "scan" ? average(tab, ByScan()) : AVG == "none" ? tab : error("AVG must be scan or none")
stamp("loaded $(length(measurement(tab))) visibilities")

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
grid = g2 ⊗ frames(Ti, collect(edges[1:(end - 1)]), collect(edges[2:end]))

crescent(ξ) = modify(RingTemplate(RadialGaussian(0.15), AzimuthalCosine(0.6, ξ)), Stretch(μas2rad(21.0)))
pas = range(0.0, π / 2; length = NT)
meanimg = stack(ξ -> baseimage(intensitymap(crescent(ξ), g2)), pas)
logmean = log.(meanimg .+ 1.0e-3 * maximum(meanimg))

function movie_sky(θ, grid, meta)
    (; c, σimg) = θ
    δ = cat(map(ck -> reshape(ck.params, size(ck.params)..., 1), c)...; dims = 3)
    e = exp.(meta.logmean .+ σimg .* δ)
    img = meta.ftot .* e ./ sum(e; dims = (1, 2))
    return ContinuousImage(IntensityMap(img, grid), BSplinePulse{3}())
end

cprior = corr_image_prior(g2, tab)
prior = (c = ntuple(_ -> cprior, NT), σimg = VLBIExponential(0.5))
skym = SkyModel(movie_sky, prior, grid; metadata = (; logmean, ftot = 0.6))

## Instrument: complex gain per station and scan

lgp = IIDSitePrior(ScanSeg(), VLBIGaussian(0.0, 0.2))
gpp = IIDSitePrior(ScanSeg(), VLBIGaussian(0.0, 1.0))
G = Jones((x, c) -> exp(complex(x.lg, x.gp)); lg = ArrayPrior(lgp), gp = ArrayPrior(gpp; refant = SensitiveReference(0.0)))
intm = InstrumentModel(G)

## Simulated data on the real coverage

rng = Xoshiro(17)
post_sim = VLBIPosterior(skym, intm, tab; admode = nothing)
θ0 = prior_sample(rng, post_sim)
θtrue = (sky = merge(θ0.sky, (σimg = 0.3,)), instrument = θ0.instrument)
dvis = only(simulate_observation(rng, post_sim, θtrue))
post_cpu = VLBIPosterior(skym, intm, dvis; admode = nothing)
tcpu = asflat(post_cpu)
x0 = prior_sample(Xoshiro(41), tcpu)
stamp("posterior: $(length(measurement(dvis))) visibilities, $NT frames of $(NPIX)×$NPIX, $(dimension(tcpu)) parameters")

fgrad(x, tp) = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Const(Base.Fix1(logdensityof, tp)), x)[1]

function bench(f, n)
    f()
    ts = [@elapsed(f()) for _ in 1:n]
    return (; median = median(ts), min = minimum(ts))
end

results = Dict{String, Any}()

## CPU reference (plain Julia, NFFT on the host, Enzyme on the CPU)

ld_cpu = gr_cpu = nothing
if "cpu" in MODES || "single" in MODES || "shard" in MODES
    tc = @elapsed ld_cpu = logdensityof(tcpu, x0)
    tg = @elapsed gr_cpu = fgrad(x0, tcpu)
    stamp("CPU first calls: logdensity $(fmt(tc)), gradient $(fmt(tg)) (includes Julia/Enzyme compilation); logdensity = $ld_cpu")
end
if "cpu" in MODES
    bl = bench(() -> logdensityof(tcpu, x0), NEVAL)
    bg = bench(() -> fgrad(x0, tcpu), max(3, NEVAL ÷ 4))
    results["cpu"] = (; compile_ld = NaN, compile_grad = NaN, ld = bl, grad = bg)
    stamp("CPU: logdensity $(fmt(bl.median)) (min $(fmt(bl.min))), gradient $(fmt(bg.median)) (min $(fmt(bg.min)))")
end

# Compile, check against the CPU, and time the log density and gradient of `tdev` at `xr`.
function run_reactant(label, tdev, xr)
    tcl = @elapsed ldc = @compile sync = true logdensityof(tdev, xr)
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

## One device

if "single" in MODES
    tsingle = asflat(todevice(post_cpu, Reactant.Sharding.NoSharding()))
    results["single"], _, _ = run_reactant("1 device", tsingle, Reactant.to_rarray(x0))
    tsingle = nothing
    GC.gc()
end

## All devices, sharded over the movie frames

# The device arrays that `x` holds.
function devicearrays(x, found = Any[], seen = IdDict{Any, Nothing}())
    if x isa Reactant.AbstractConcreteArray
        push!(found, x)
    elseif x isa Array
        isbitstype(eltype(x)) || foreach(y -> devicearrays(y, found, seen), x)
    elseif !(isbits(x) || x isa Union{Module, Type, Symbol, AbstractString} || haskey(seen, x))
        seen[x] = nothing
        foreach(i -> isdefined(x, i) && devicearrays(getfield(x, i), found, seen), 1:fieldcount(typeof(x)))
    end
    return found
end

function collectives(hlo)
    ops = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute", "while")
    return NamedTuple{Symbol.(replace.(ops, "-" => "_"))}(map(op -> count(op * "(", hlo) + count(op * "-start(", hlo), ops))
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
    mesh = Reactant.Sharding.Mesh(reshape(DEVS, NDEV), (:d,))
    layout = ShardLayout(mesh; Ti = :d)
    tp = @elapsed post_dev = todevice(post_cpu, layout)
    stamp("sharded posterior placed in $(fmt(tp))")

    arrs = devicearrays(post_dev)
    nsh = count(Reactant.Sharding.is_sharded, arrs)
    obs = only(post_dev.lklhds).obs
    slices = obs.sharding.device_to_array_slices
    println("device arrays in the posterior: $(length(arrs)), all with a sharding: $(nsh == length(arrs))")
    println("data on the devices: size $(size(obs)) (host $(length(measurement(dvis))) visibilities, padded per device)")
    for (i, s) in enumerate(slices)
        println("  device $i holds rows $(s[1]) of the data")
    end

    tdev = asflat(post_dev)
    xr = Reactant.to_rarray(x0; sharding = Reactant.Sharding.Replicated(mesh))
    results["shard"], ldc, gc = run_reactant("$NDEV devices", tdev, xr)

    if HLO
        hl = string(Reactant.@code_xla logdensityof(tdev, xr))
        hg = string(Reactant.@code_xla fgrad(xr, tdev))
        println("partitioned HLO collectives, logdensity: ", collectives(hl))
        println("partitioned HLO collectives, gradient:   ", collectives(hg))
        nv = size(obs, 1)
        println("arrays of the per-device data length [$(nv ÷ NDEV), ...] in the partitioned gradient HLO: $(count("[$(nv ÷ NDEV),", hg) + count("[$(nv ÷ NDEV)]", hg)); of the full length [$nv, ...]: $(count("[$nv,", hg) + count("[$nv]", hg))")
        write(joinpath(@__DIR__, "out_logdensity.hlo"), hl)
        write(joinpath(@__DIR__, "out_gradient.hlo"), hg)
    end

    NOPT > 0 && run_adam(tdev, ldc, x0, mesh, NOPT)
end
println("\nSUMMARY (median of $NEVAL calls; NPIX = $NPIX, NT = $NT, $(length(measurement(dvis))) visibilities, $(dimension(tcpu)) parameters)")
for k in ("cpu", "single", "shard")
    haskey(results, k) || continue
    r = results[k]
    name = k == "cpu" ? "CPU (no Reactant)" : k == "single" ? "Reactant, 1 device" : "Reactant, $NDEV devices"
    err = hasproperty(r, :rld) ? ", rel. err ld $(round(r.rld; sigdigits = 2)), grad $(round(r.rgr; sigdigits = 2))" : ""
    println(rpad(name, 22), " compile ld ", lpad(fmt(r.compile_ld), 9), ", grad ", lpad(fmt(r.compile_grad), 9),
        " | logdensity ", lpad(fmt(r.ld.median), 9), " | gradient ", lpad(fmt(r.grad.median), 9), err)
end
stamp("done")
