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
        merge(J.layout, (; stacks)), J.iterations, _slots(points, to)[J.keep], entry(J.entry1), entry(J.entry2)
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
# A single index fails in the broadcast form (`Float64(::TracedRArray)`), e.g. a TrackSeg
# offset referenced to one antenna, so it uses a scalar store. `inds` is a host vector, so
# the length checks resolve at trace time.
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

# Log-depth (Hillis–Steele) scan of the affine recurrence of the generic method, using
# `(a₂, c₂) ∘ (a₁, c₁) = (a₂a₁, a₂c₁ + c₂)`. The `while` loop runs at trace time and unrolls
# into ⌈log₂ n⌉ rounds of dense operations; a sequential recurrence would trace to a
# `stablehlo.while` that Reactant does not vectorize. Zeros in `a` reset the carry between
# concatenated chains, with exact gradients. Mutates both arguments.
#
# A Blelloch scan is not used: Enzyme-MLIR mis-accumulates the cotangents of its
# differently-sized subslices. `cumprod` closed forms are unstable. If Reactant provides an
# associative-scan primitive, this method can be deleted.
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

# Traced parameters walk batched chain groups: a per-chain walk would launch one kernel per
# chain. The host walks chains one at a time.
@inline function VLBIInstrumentModels._walk_units(d::VLBIInstrumentModels.GaussMarkovChainDist, ::Reactant.AnyTracedRArray)
    return VLBIInstrumentModels.chaingroups(d)
end

end
