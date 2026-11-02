export simulate_observation

"""
    simulate_observation([rng::AbstractRNG], vis, J, x, t::VisTable; add_thermal_noise = true)

The table `t` with its measurements replaced by the model: the sky's model visibilities `vis`
over the records of `t`, corrupted by the instrument model `J` bound to `t` with
[`observe`](@ref), at the instrument parameters `x`. With `add_thermal_noise`, Gaussian noise
with the weights of `t` is added. Values without a measurement in `t` stay unmeasured.
"""
function simulate_observation(rng::AbstractRNG, vis, J, x, t::VisTable; add_thermal_noise = true)
    return _withmodel(t, _observedvalues(rng, apply_instrument(vis, J, x), t, add_thermal_noise))
end
simulate_observation(vis, J, x, t::VisTable; add_thermal_noise = true) =
    simulate_observation(Random.default_rng(), vis, J, x, t; add_thermal_noise)

# The dense storage of the values `v` of the table `t`, with thermal noise when `noise`.
function _observedvalues(rng, v, t::VisTable, noise)
    s = _storage(baseimage(v))
    noise || return s
    return rand(rng, _noiselikelihood(eltype(measurement(t)), s, _dense(weight(t))))
end

_storage(μ::FieldDimArray) = parent(μ)
_storage(μ) = μ

_noiselikelihood(::Type{<:StaticArraysCore.StaticMatrix}, vis, w) = CoherencyLikelihood(vis, w)
_noiselikelihood(::Type, vis, w) = ComplexVisLikelihood(vis, w)

# `d` with its measurements replaced by those formed from the model visibilities `vis`.
function _withmodel(t::VisTable, vis)
    m = _asmeasured(eltype(measurement(t)), vis)
    w = _dense(weight(t))
    return VisTable(t.pol, IntensityMap(_elements(t.pol, _zeroed(m, w)), domain(t)), weight(t), stations(t))
end
function _withmodel(ct::ClosureTable, vis)
    x = _closurevalue.(Ref(ct.product), vis[_measuredvalues(getfield(ct, :vis))])
    m = IntensityMap(getfield(ct, :design) * x, domain(ct))
    return ClosureTable(ct.product, m, getfield(ct, :vis), getfield(ct, :design), getfield(ct, :covariance))
end

_asmeasured(::Type{<:Real}, vis) = abs.(vis)
_asmeasured(::Type, vis) = vis

# Values without a measurement are stored as 0.
_zeroed(m::AbstractArray{<:Number}, w) = ifelse.(w .> 0, m, zero(eltype(m)))
function _zeroed(m::AbstractArray{<:StaticArraysCore.StaticMatrix}, w)
    d = reshape(reinterpret(eltype(eltype(m)), collect(m)), 2, 2, size(m)...)
    return ifelse.(w .> 0, permutedims(d, (3, 4, 1, 2)), zero(eltype(d)))
end
