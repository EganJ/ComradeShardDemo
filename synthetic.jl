# Synthetic coverage for the stations, source and epoch of a real table.

using Dates

# Greenwich mean sidereal time, in radians, `t` hours after midnight UTC of `epoch`.
function gmst(epoch, t)
    jd = datetime2julian(DateTime(epoch)) + t / 24
    return mod(18.697374558 + 24.06570982441908 * (jd - 2451545.0), 24) * π / 12
end

# Source elevation, in radians, at the geocentric position `r` (meters) for the Greenwich
# hour angle `H` and declination `δ`.
function elevation(r, H, δ)
    lat = atan(r[3], hypot(r[1], r[2]))
    h = H + atan(r[2], r[1])
    return asin(sin(lat) * sin(δ) + cos(lat) * cos(δ) * cos(h))
end

# The baseline `r1 - r2` projected on the sky, in meters, as stored in uvfits files.
function baseline_uv(r1, r2, H, δ)
    B = r1 - r2
    return (
        sin(H) * B[1] + cos(H) * B[2],
        -sin(δ) * cos(H) * B[1] + sin(δ) * sin(H) * B[2] + cos(δ) * B[3],
    )
end

"""
    read_array(path)

The stations of an ehtim array file: name, geocentric position in meters, `Mount(FR_PAR,
FR_EL)`, circular feeds, and `FR_OFF` (degrees) as the angle of both feeds.
"""
function read_array(path)
    rows = [split(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    num(r, i) = parse(Float64, r[i])
    return (
        name = [Symbol(r[1]) for r in rows],
        position = [[num(r, 2), num(r, 3), num(r, 4)] for r in rows],
        mount = [Mount(num(r, 7), num(r, 8)) for r in rows],
        feeds = [(RPol(), LPol()) for _ in rows],
        feed_angles = [(deg2rad(num(r, 9)), deg2rad(num(r, 9))) for r in rows],
    )
end

# Per-station noise factors `f` with `σ ≈ f[a] * f[b]` on baseline `a`-`b`, fitted by least
# squares to the log of each baseline's median noise in `tab`, by station name.
function station_noise(tab)
    ai, d = Comrade.info(tab), domain(tab)
    σ = vec(Comrade.noise(tab))
    pairs = unique(zip(d.ant1, d.ant2))
    A = zeros(length(pairs), length(ai.stations.name))
    y = [log(median(σ[(d.ant1 .== a) .& (d.ant2 .== b)])) for (a, b) in pairs]
    for (k, (a, b)) in enumerate(pairs)
        A[k, a] = A[k, b] = 1
    end
    used = findall(vec(any(!iszero, A; dims = 1)))
    return Dict(zip(ai.stations.name[used], exp.(A[:, used] \ y)))
end

"""
    synthetic_table(tab; ndays, tint, nif, nchan, stations = nothing, aliases = Dict(),
                    elmin = 10, scanlen = 10, scangap = 2)

A Stokes I table with the source and epoch of `tab` and zero visibilities.

Stations: those of `tab`, or `stations` (columns as in [`read_array`](@ref)).

Records: every baseline that appears in `tab`, or every pair of `stations`, at integrations
of `tint` seconds, in scans of `scanlen` minutes separated by `scangap` minutes, over `ndays`
days from the epoch, at times when the source is at least `elmin` degrees above the horizon
at both stations.

Channels: `nif` contiguous IFs of 2 GHz centered on the frequency of `tab`, each split into
`nchan` channels.

Noise, scaled by the square root of the ratios of bandwidth and integration time: each
baseline's median noise in `tab`; with `stations`, the product of the two stations' factors
from [`station_noise`](@ref), where `aliases` maps a station name to its name in `tab` and
stations absent from `tab` get the median factor.

Returns the table and the IF edges `(lo, hi)` in Hz.
"""
function synthetic_table(
        tab; ndays, tint, nif, nchan, stations = nothing, aliases = Dict{Symbol, Symbol}(),
        elmin = 10.0, scanlen = 10.0, scangap = 2.0
    )
    ai = Comrade.info(tab)
    d = domain(tab)
    length(ai.channels) == 1 || throw(ArgumentError("the template table must have one channel, not $(length(ai.channels))"))
    st = something(stations, ai.stations)
    pos = st.position
    α, δ = deg2rad(ai.source.ra), deg2rad(ai.source.dec)

    if isnothing(stations)
        σ0 = vec(Comrade.noise(tab))
        pairs = unique(zip(d.ant1, d.ant2))
        σpair = Dict(p => median(σ0[(d.ant1 .== p[1]) .& (d.ant2 .== p[2])]) for p in pairs)
    else
        fs = station_noise(tab)
        fmed = median(collect(values(fs)))
        f = [get(fs, get(aliases, n, n), fmed) for n in st.name]
        pairs = [(a, b) for a in eachindex(st.name) for b in (a + 1):lastindex(st.name)]
        σpair = Dict((a, b) => f[a] * f[b] for (a, b) in pairs)
    end
    tint0 = median(diff(sort!(unique(d.Ti)))) * 3600

    ν0 = only(ai.channels.frequency)
    ifwidth = 2.0e9
    iflo = [ν0 + (k - (nif + 1) / 2 - 0.5) * ifwidth for k in 1:nif]
    width = ifwidth / nchan
    freq = vec([lo + (j - 0.5) * width for j in 1:nchan, lo in iflo])
    scale = sqrt(only(ai.channels.width) / width * tint0 / tint)

    nint = round(Int, scanlen * 60 / tint)
    nint >= 1 || throw(ArgumentError("a scan of $scanlen min holds no integration of $tint s"))
    cycle = (scanlen + scangap) / 60
    u, v, Ti, ant1, ant2, σb = Float64[], Float64[], Float64[], Int[], Int[], Float64[]
    scans = (start = Float64[], stop = Float64[])
    for day in 0:(ndays - 1), k in 0:(floor(Int, 24 / cycle) - 1)
        t0 = 24day + k * cycle
        n0 = length(u)
        for j in 1:nint
            t = t0 + (j - 0.5) * tint / 3600
            H = gmst(ai.epoch, t) - α
            up = map(r -> elevation(r, H, δ) >= deg2rad(elmin), pos)
            for (a1, a2) in pairs
                up[a1] && up[a2] || continue
                uu, vv = baseline_uv(pos[a1], pos[a2], H, δ)
                push!(u, uu); push!(v, vv); push!(Ti, t); push!(ant1, a1); push!(ant2, a2)
                push!(σb, σpair[(a1, a2)] * scale)
            end
        end
        if length(u) > n0
            push!(scans.start, t0)
            push!(scans.stop, t0 + nint * tint / 3600)
        end
    end
    isempty(u) && throw(ArgumentError("no baseline sees the source above $elmin degrees"))

    days = floor.(Int, scans.start ./ 24)
    tracks = (
        start = [minimum(scans.start[days .== k]) for k in unique(days)],
        stop = [maximum(scans.stop[days .== k]) for k in unique(days)],
    )
    info = ArrayInfo(;
        stations = st, channels = (; frequency = freq, width = fill(width, length(freq))),
        scans, tracks, source = ai.source, epoch = ai.epoch,
    )
    σ = repeat(σb, 1, length(freq))
    V = zeros(ComplexF64, size(σ))
    return VisTable(IPol(), V, σ; u, v, Ti, ant1, ant2, info), (iflo, iflo .+ ifwidth)
end
