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
    synthetic_table(tab; ndays, tint, nif, nchan, elmin = 10, scanlen = 10, scangap = 2)

A Stokes I table with the stations, source and epoch of `tab` and zero visibilities.

Records: every baseline that appears in `tab`, at integrations of `tint` seconds, in scans
of `scanlen` minutes separated by `scangap` minutes, over `ndays` days from the epoch, at
times when the source is at least `elmin` degrees above the horizon at both stations.

Channels: `nif` contiguous IFs of 2 GHz centered on the frequency of `tab`, each split into
`nchan` channels.

Noise: each baseline's median noise in `tab`, scaled by the square root of the ratios of
bandwidth and integration time.

Returns the table and the IF edges `(lo, hi)` in Hz.
"""
function synthetic_table(tab; ndays, tint, nif, nchan, elmin = 10.0, scanlen = 10.0, scangap = 2.0)
    ai = Comrade.info(tab)
    d = domain(tab)
    length(ai.channels) == 1 || throw(ArgumentError("the template table must have one channel, not $(length(ai.channels))"))
    pos = ai.stations.position
    α, δ = deg2rad(ai.source.ra), deg2rad(ai.source.dec)

    σ0 = vec(Comrade.noise(tab))
    pairs = unique(zip(d.ant1, d.ant2))
    σpair = Dict(p => median(σ0[(d.ant1 .== p[1]) .& (d.ant2 .== p[2])]) for p in pairs)
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
        stations = ai.stations, channels = (; frequency = freq, width = fill(width, length(freq))),
        scans, tracks, source = ai.source, epoch = ai.epoch,
    )
    σ = repeat(σb, 1, length(freq))
    V = zeros(ComplexF64, size(σ))
    return VisTable(IPol(), V, σ; u, v, Ti, ant1, ant2, info), (iflo, iflo .+ ifwidth)
end
