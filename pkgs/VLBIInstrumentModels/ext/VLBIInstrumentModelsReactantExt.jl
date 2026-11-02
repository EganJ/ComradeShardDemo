module VLBIInstrumentModelsReactantExt

using VLBIInstrumentModels
import VLBIInstrumentModels as VIM
using ComradeBase: ComradeBase, ReactantEx, ShardLayout, StructuredDomain, todevice
using Reactant
using StructArrays

"""
    todevice(J, d::StructuredDomain)
    todevice(J, sharding::Reactant.Sharding.AbstractSharding)

Return a copy of the instrument model `J`, bound to a table by [`observe`](@ref), with its
arrays placed on Reactant devices to act on model visibilities over `d`: the points of the
table placed by `todevice(domain, layout)`, such as the visibility domain of a sharded dual
domain. Each value's station cells follow the points of `d`, padding slots reading cell 1;
so do the self-calibration's residual entries and the slots of its measured values. The
per-cell coordinates, values and responses, the instrument terms and the self-calibration
data and weights are replicated over the mesh of `d`. A model bound to several fields takes
a `NamedTuple` of domains, one per field.

With a `sharding`, every array is placed with it.
"""
ComradeBase.todevice(J::VIM.AbstractInstrumentModel, d::StructuredDomain) = _shardinstrument(J, d)
ComradeBase.todevice(J::VIM.AbstractInstrumentModel, s::Reactant.Sharding.AbstractSharding) = _shardinstrument(J, s)
ComradeBase.todevice(J::VIM.FieldInstruments, ds::NamedTuple) = VIM.FieldInstruments(map(todevice, J.fields, ds))

_shardinstrument(J::VIM.ObservedIdealInstrument, _) = J
_shardinstrument(J::VIM.FieldInstruments, s) = VIM.FieldInstruments(map(j -> todevice(j, s), J.fields))

function _shardinstrument(J::VIM.ObservedInstrumentModel, to)
    return VIM.ObservedInstrumentModel(
        _shardterms(J.instrument, to), J.refbasis, _shardcells(J.layout, to), J.pol, VIM.corrected(J)
    )
end

_shardterms(J, to) = _replicate(J, to)
_shardterms(J::VIM._StationGroups, to) = VIM._StationGroups(_replicate(J.terms, to), J.cells)

function _shardcells(L::VIM.CellLayout, to)
    cell(c) = _pervalue(c, L.points, to, one(eltype(c)))
    coords = StructArray(map(c -> _replicate(c, to), StructArrays.components(L.coords)))
    response = isnothing(L.response) ? nothing : _replicate(L.response, to)
    return VIM.CellLayout(coords, cell(L.cell1), cell(L.cell2), _replicate(L.entries, to), response, L.points)
end

function _shardinstrument(J::VIM.ObservedSelfCalInstrumentModel, to)
    points = J.model.layout.points
    entry(e) = _pervalue(e, points, to, one(eltype(e)))
    stacks = map(st -> merge(st, (; eye = _replicate(st.eye, to))), J.layout.stacks)
    return VIM.ObservedSelfCalInstrumentModel(
        _shardinstrument(J.model, to), J.phase,
        map(w -> _replicate(w, to), J.wdata), map(w -> _replicate(w, to), J.weight),
        merge(J.layout, (; stacks)), J.squarings, _slots(points, to)[J.keep], entry(J.entry1), entry(J.entry2)
    )
end

function _layout(d::StructuredDomain)
    ex = ComradeBase.executor(d)
    ex isa ReactantEx && ex.layout isa ShardLayout || throw(
        ArgumentError("the domain is not sharded; place it with `todevice(domain, layout)` first")
    )
    return ex.layout
end

_replicate(x, d::StructuredDomain) = todevice(x, Reactant.Sharding.Replicated(_layout(d).mesh))
_replicate(x, s) = todevice(x, s)

# The values `v` over the table's `points`, laid out like `d`; padding slots hold `pad`.
_pervalue(v, points, d::StructuredDomain, pad) = baseimage(todevice(IntensityMap(v, StructuredDomain(points)), d; pad))
_pervalue(v, points, s, pad) = todevice(v, s)

_slots(points, d::StructuredDomain) = ComradeBase._slots(points, d)
_slots(points, s) = LinearIndices(map(length, points))

# The indexed assignment lowers to `stablehlo.scatter`, but its bounds check iterates the
# index vector with scalar `getindex`, which is disallowed while tracing. Allowing scalar
# indexing here lets the trace through. See https://github.com/EnzymeAD/Reactant.jl/issues/2960.
#
# `y[inds] .= vals` lowers to a `stablehlo.scatter` — the fast, vectorized ("raised")
# path — for length > 1, but its single-element case hits a Reactant bug that tries to
# assign a 1-element traced array into a scalar slot (`Float64(::TracedRArray)`), e.g. a
# TrackSeg offset referenced to one antenna has exactly one fixed index. `inds` is a
# concrete host vector, so the length branches are resolved at trace time (no traced
# branch): empty is a no-op and only the length-1 case falls back to a scalar store.
# See EnzymeAD/Reactant.jl#2960.
@inline function VLBIInstrumentModels.scatter_values!(y::Reactant.AnyTracedRArray, inds, vals)
    isempty(inds) && return y
    Reactant.@allowscalar begin
        if length(inds) == 1
            y[inds[begin]] = vals[begin]
        else
            y[inds] .= vals
        end
    end
    return y
end

# Log-depth (Hillis-Steele) scan for the recurrence in the generic method. Affine
# maps compose associatively — `(a₂,c₂) ∘ (a₁,c₁) = (a₂a₁, a₂c₁ + c₂)` — so after
# the round with shift `s` entries `1:s` are final, and one slice-multiply-add folds
# the partial scan `s` places into the tail, doubling the resolved prefix. The
# `while` runs on the HOST at trace time and unrolls into ⌈log₂ n⌉ rounds of dense
# ops (~13 rounds for IntegSeg-length chains, hard-capped at 64) — it exists because
# Reactant's loop raising never vectorizes a loop-carried recurrence, so the
# sequential form would serialize into a `stablehlo.while` in every logdensity and
# gradient evaluation. Composition is plain mul/add, so the exact zeros the coloring
# places in `a` at chain-opening and fixed-left points reset the carry across the
# batched chain concatenation with exact gradients. Mutates its arguments
# (trace-local staging temps at both call sites).
#
# NB: the prettier work-efficient formulations are blocked upstream — the recursive
# odd-even (Blelloch) scan needs differently-sized subslices of one array, whose
# cotangents Enzyme-MLIR currently mis-accumulates (`stablehlo.add` shape mismatch),
# and `cumprod`-based closed forms are unstable with a broken Enzyme gradient. If
# Reactant ships an associative-scan primitive, delete this method — the generic
# sequential definition then serves both backends.
function VLBIInstrumentModels.affine_scan!(a::Reactant.AnyTracedRArray, c::Reactant.AnyTracedRArray)
    n = length(c)
    s = 1
    while s < n
        at = a[(s + 1):n]
        c[(s + 1):n] = at .* c[1:(n - s)] .+ c[(s + 1):n]
        a[(s + 1):n] = at .* a[1:(n - s)]
        s <<= 1
    end
    return c
end

# Batched chain groups are worth their per-point parameter gather only where a per-chain
# walker would cost a kernel launch per chain — i.e. exactly when the parameter vector is
# traced. On the host the per-chain form wins (see `VLBIInstrumentModels._walk_units`).
@inline function VLBIInstrumentModels._walk_units(d::VLBIInstrumentModels.GaussMarkovChainDist, ::Reactant.AnyTracedRArray)
    return VLBIInstrumentModels.chaingroups(d)
end

end
