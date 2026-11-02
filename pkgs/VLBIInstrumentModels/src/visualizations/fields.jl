export DataField, datafield

"""
    DataField(value, label; title = nothing)

A quantity of the rows of [`datarows`](@ref), for plotting: `value(row)` gives it for one
row, `label` names it on an axis (`nothing` names it by the table's measurements), and
`title`, when given, maps the values plotted on one axis to that axis's title.
"""
struct DataField{F, T}
    value::F
    label::Union{Nothing, String}
    title::T
end
DataField(value, label; title = nothing) = DataField(value, isnothing(label) ? nothing : string(label), title)

(f::DataField)(row) = f.value(row)

"""
    datafield(name)

The [`DataField`](@ref) called `name`: a `Symbol`, or a dimension such as `Ti`. A function
`f` is the field with values `f(row)`. The fields of visibility and closure rows are

  - `:U`, `:V`: the baseline coordinates (λ)
  - `:uvdist`: the projected baseline length, the longest one of a closure (λ)
  - `:Ti`, `:Fr`: the time and frequency
  - `:sites` (visibilities), `:stations` (closures)
  - `:measurement`, `:noise`
  - `:measwnoise`: the measurement with its noise, plotted with error bars
  - `:amp`, `:phase`, `:snr`: the measurement's amplitude, phase and signal-to-noise ratio
  - `:res`: the measurement over its noise, the normalized residual of a residual table,
    with the reduced χ² of each axis as its title

Define another with a method `VLBIInstrumentModels.datafield(::Val{:name})`.
"""
datafield(name::Symbol) = datafield(Val(name))
datafield(::Type{D}) where {D <: DD.Dimension} = datafield(DD.name(D))
datafield(f::DataField) = f
datafield(f) = DataField(f, string(f))
datafield(::Val{N}) where {N} = throw(
    ArgumentError("no data field `:$N`; define one with a method `VLBIInstrumentModels.datafield(::Val{:$N})`")
)

_rowU(r) = r.U
_rowV(r) = r.V

datafield(::Val{:U}) = DataField(_rowU, "u (λ)")
datafield(::Val{:V}) = DataField(_rowV, "v (λ)")
datafield(::Val{:uvdist}) = DataField(r -> r.uvdist, "Projected Baseline Distance (λ)")
datafield(::Val{:Ti}) = DataField(r -> r.Ti, "Time (UTC)")
datafield(::Val{:Fr}) = DataField(r -> r.Fr, "Frequency (Hz)")
datafield(::Val{:sites}) = DataField(r -> r.sites, "Baseline")
datafield(::Val{:stations}) = DataField(r -> r.stations, "Stations")
datafield(::Val{:measurement}) = DataField(r -> r.measurement, nothing)
datafield(::Val{:noise}) = DataField(r -> r.noise, "Noise (Jy)")
datafield(::Val{:measwnoise}) = DataField(r -> _withnoise(r.measurement, r.noise), nothing)
datafield(::Val{:amp}) = DataField(r -> abs.(r.measurement), "Visibility Amplitude (Jy)")
datafield(::Val{:phase}) = DataField(r -> angle.(r.measurement), "Visibility Phase (rad)")
datafield(::Val{:snr}) = DataField(r -> abs.(r.measurement) ./ r.noise, "SNR")
datafield(::Val{:res}) = DataField(r -> r.measurement ./ r.noise, "Normalized Residual"; title = _chi2title)

_withnoise(m::Real, n) = Measurements.measurement(m, n)
_withnoise(m::Complex, n) = complex(Measurements.measurement(real(m), n), Measurements.measurement(imag(m), n))
_withnoise(m::AbstractMatrix, n::AbstractMatrix) = map(_withnoise, m, n)

# The reduced χ² of normalized residuals `r`: Σ |r|² over the measured values, per real datum.
function _chi2(r)
    z = filter(!isnan, r)
    return sum(abs2, z) / (eltype(z) <: Complex ? 2 * length(z) : length(z))
end
_chi2title(r) = Printf.@sprintf("χ² = %.2f", _chi2(r))
