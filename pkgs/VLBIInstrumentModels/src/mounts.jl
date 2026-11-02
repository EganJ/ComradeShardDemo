export AbstractMount, Mount, MountAltAz, MountEquatorial, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL, MountXY, MountOrbiting, MountOther, MountRotator, MountHALCA,
    feed_rotation, axis_offset

"""
    AbstractMount

How a station carries its feed, which sets how the feed rotates against the sky as the station
tracks a source. [`Mount`](@ref) covers the mounts whose rotation is a fixed combination of
the parallactic angle, elevation and azimuth; [`MountXY`](@ref), [`MountOrbiting`](@ref),
[`MountOther`](@ref), [`MountRotator`](@ref) and [`MountHALCA`](@ref) are mounts whose rotation
is not such a combination. The names follow the MSv4 `mount` field.
"""
abstract type AbstractMount end

"""
    Mount(parallactic, elevation, azimuth = zero(elevation); offset = (0.0, 0.0, 0.0))

A mount whose feed rotates by `parallactic · χ + elevation · E + azimuth · A` against the sky,
with `χ` the parallactic angle, `E` the elevation and `A` the azimuth. `offset` is the
displacement between the mount's axes in meters (see [`axis_offset`](@ref)). The common mounts
have constructors: [`MountAltAz`](@ref), [`MountEquatorial`](@ref), [`MountNasmythR`](@ref),
[`MountNasmythL`](@ref), [`MountBWGR`](@ref), [`MountBWGL`](@ref).
"""
struct Mount{A, B, C, O} <: AbstractMount
    parallactic::A
    elevation::B
    azimuth::C
    offset::O
end

Mount(parallactic, elevation, azimuth = zero(elevation); offset = ntuple(Returns(0.0), 3)) =
    Mount(parallactic, elevation, azimuth, offset)

_coefficients(m::Mount) = (m.parallactic, m.elevation, m.azimuth)
Base.:(==)(a::Mount, b::Mount) = _coefficients(a) == _coefficients(b) && a.offset == b.offset
Base.hash(m::Mount, h::UInt) = hash((_coefficients(m)..., m.offset), hash(:Mount, h))

"""
    MountAltAz(offset = (0.0, 0.0, 0.0))

An alt-azimuth mount: the feed rotates by the parallactic angle.
"""
MountAltAz(offset = ntuple(Returns(0.0), 3)) = Mount(1.0, 0.0, 0.0, offset)

"""
    MountEquatorial(offset = (0.0, 0.0, 0.0))

An equatorial mount: the feed does not rotate against the sky.
"""
MountEquatorial(offset = ntuple(Returns(0.0), 3)) = Mount(0.0, 0.0, 0.0, offset)

"""
    MountNasmythR(offset = (0.0, 0.0, 0.0))

A right Nasmyth mount: the feed rotates by the parallactic angle plus the elevation.
"""
MountNasmythR(offset = ntuple(Returns(0.0), 3)) = Mount(1.0, 1.0, 0.0, offset)

"""
    MountNasmythL(offset = (0.0, 0.0, 0.0))

A left Nasmyth mount: the feed rotates by the parallactic angle minus the elevation.
"""
MountNasmythL(offset = ntuple(Returns(0.0), 3)) = Mount(1.0, -1.0, 0.0, offset)

"""
    MountBWGR(offset = (0.0, 0.0, 0.0))

A right beam-waveguide mount: the feed rotates by `χ + E − A` (Dodson & Rioja,
arXiv:2210.13381).
"""
MountBWGR(offset = ntuple(Returns(0.0), 3)) = Mount(1.0, 1.0, -1.0, offset)

"""
    MountBWGL(offset = (0.0, 0.0, 0.0))

A left beam-waveguide mount: the feed rotates by `χ − E + A`.
"""
MountBWGL(offset = ntuple(Returns(0.0), 3)) = Mount(1.0, -1.0, 1.0, offset)

"""
    MountXY(offset = (0.0, 0.0, 0.0))

A mount with a horizontal primary axis. Its feed rotation is not a combination of parallactic
angle, elevation and azimuth.
"""
struct MountXY{C} <: AbstractMount
    offset::C
end

"""
    MountOrbiting(offset = (0.0, 0.0, 0.0))

The mount of a station in orbit, whose feed rotation depends on the spacecraft's attitude.
"""
struct MountOrbiting{C} <: AbstractMount
    offset::C
end

"""
    MountOther(offset = (0.0, 0.0, 0.0))

A mount of unknown type: nothing about its feed rotation is known.
"""
struct MountOther{C} <: AbstractMount
    offset::C
end

"""
    MountRotator(offset = (0.0, 0.0, 0.0))

An alt-azimuth mount with a feed rotator, whose angle depends on how the rotator is driven.
"""
struct MountRotator{C} <: AbstractMount
    offset::C
end

"""
    MountHALCA(offset = (0.0, 0.0, 0.0))

The mount of HALCA, the VSOP space-VLBI antenna, whose orientation follows its own model.
"""
struct MountHALCA{C} <: AbstractMount
    offset::C
end

for T in (:MountXY, :MountOrbiting, :MountOther, :MountRotator, :MountHALCA)
    @eval $T() = $T(ntuple(Returns(0.0), 3))
end

"""
    axis_offset(m::AbstractMount)

The displacement between the mount's two rotation axes, in meters, along each axis of the
station's frame; zero for a mount whose axes intersect.
"""
axis_offset(m::AbstractMount) = m.offset

"""
    feed_rotation(m::AbstractMount, parallactic, elevation, azimuth)

The angle, in radians, by which the mount `m` rotates its feed against the sky when the source
is at the given parallactic angle, elevation and azimuth (radians).
"""
feed_rotation(m::Mount, parallactic, elevation, azimuth) =
    m.parallactic * parallactic + m.elevation * elevation + m.azimuth * azimuth
function feed_rotation(m::AbstractMount, parallactic, elevation, azimuth)
    throw(
        ArgumentError(
            "the feed rotation of a $(nameof(typeof(m))) is not a function of parallactic angle, elevation and azimuth"
        )
    )
end
