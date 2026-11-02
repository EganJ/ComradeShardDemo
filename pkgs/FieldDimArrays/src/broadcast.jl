"""
    FieldDimArrayStyle{N}

The broadcast style of an `N`-dim `FieldDimArray`. It wins over `DefaultArrayStyle`. An
out-of-place broadcast whose result element type `E` satisfies `isfieldelement(E)` returns a
`FieldDimArray{E}` over new storage allocated like the parent of the first
`FieldDimArray` argument; other results are allocated with `similar` of that parent.
"""
struct FieldDimArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
FieldDimArrayStyle{M}(::Val{N}) where {M, N} = FieldDimArrayStyle{N}()

Base.BroadcastStyle(::Type{<:FieldDimArray{T, N}}) where {T, N} = FieldDimArrayStyle{N}()

"""
    componentwise(parent::AbstractArray) -> Bool

Whether broadcasts involving `FieldDimArray`s over parents like `parent` run one broadcast
per component over the component slabs instead of one pass over the elements. False by
default; array types that cannot index elements one at a time (such as traced arrays) return
true.
"""
componentwise(::AbstractArray) = false

function Base.similar(bc::Broadcast.Broadcasted{<:FieldDimArrayStyle}, ::Type{E}) where {E}
    p = _template(bc)
    ax = axes(bc)
    if isfieldelement(E)
        storage = similar(p, componenttype(E), (ax..., map(Base.OneTo, fieldshape(E))...))
        return FieldDimArray{E, length(ax)}(storage)
    else
        return similar(p, E, ax)
    end
end

function Base.copy(bc::Broadcast.Broadcasted{<:FieldDimArrayStyle})
    componentwise(_template(bc)) && return _componentwise_copy(bc)
    return invoke(copy, Tuple{Broadcast.Broadcasted}, bc)
end
Base.copy(bc::Broadcast.Broadcasted{FieldDimArrayStyle{0}}) = invoke(copy, Tuple{Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}}, bc)

function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{<:FieldDimArrayStyle})
    return _copyto!(dest, bc)
end
function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{FieldDimArrayStyle{0}})
    return _copyto!(dest, bc)
end
Base.copyto!(dest::FieldDimArray, bc::Broadcast.Broadcasted{Nothing}) = _copyto!(dest, bc)

@inline function _copyto!(dest, bc)
    componentwise(_template(dest, bc)) && return _componentwise_copyto!(dest, bc)
    return invoke(copyto!, Tuple{AbstractArray, Broadcast.Broadcasted{Nothing}}, dest, convert(Broadcast.Broadcasted{Nothing}, bc))
end

_template(dest::FieldDimArray, bc) = parent(dest)
_template(dest, bc) = _template(bc)
_template(bc::Broadcast.Broadcasted) = _firsttemplate(bc.args...)
_firsttemplate() = nothing
_firsttemplate(x, rest...) = _ortemplate(_argtemplate(x), rest)
_ortemplate(t, rest) = t
_ortemplate(::Nothing, rest) = _firsttemplate(rest...)
_argtemplate(x::FieldDimArray) = parent(x)
_argtemplate(x::Broadcast.Broadcasted) = _template(x)
_argtemplate(x) = nothing

"""
    Regroup{L}(f)

Calls `f` with its positional arguments regrouped by `L`, a tuple holding, for each argument
of `f`, either `Val(T)` for an element type `T` (the next `ncomponents(T)` arguments are the
components of one `T`) or `Val(nothing)` (one argument).
"""
struct Regroup{L, F}
    f::F
end
Regroup{L}(f) where {L} = Regroup{L, typeof(f)}(f)
(r::Regroup{L})(xs...) where {L} = r.f(_regroup(L, xs)...)

_regroup(::Tuple{}, ::Tuple{}) = ()
function _regroup(L::Tuple, xs::Tuple)
    x, rest = _take(first(L), xs)
    return (x, _regroup(Base.tail(L), rest)...)
end
_take(::Val{nothing}, xs::Tuple) = (first(xs), Base.tail(xs))
function _take(::Val{T}, xs::Tuple) where {T}
    n = ncomponents(T)
    return fromcomponents(T, ntuple(i -> xs[i], Val(n))), ntuple(i -> xs[n + i], Val(length(xs) - n))
end

"""
    Component{K}(f)

Calls `f` and returns component `K` of its result.
"""
struct Component{K, F}
    f::F
end
Component{K}(f) where {K} = Component{K, typeof(f)}(f)
(c::Component{K})(xs...) where {K} = components(c.f(xs...))[K]

_callwith(f, xs...) = f(xs...)

_layout(::FieldDimArray{T}) where {T} = Val(T)
_layout(_) = Val(nothing)
_leaves(a::FieldDimArray{T}) where {T} = ntuple(k -> fieldview(a, k), Val(ncomponents(T)))
_leaves(x) = (x,)

_flatleaves(args::Tuple{}) = ()
_flatleaves(args::Tuple) = (_leaves(first(args))..., _flatleaves(Base.tail(args))...)

function _componentwise(bc::Broadcast.Broadcasted)
    fl = Broadcast.flatten(bc)
    return Regroup{map(_layout, fl.args)}(fl.f), _flatleaves(fl.args)
end

# Callables are passed as `Ref` arguments: Reactant cannot broadcast a callable struct that
# holds traced values.
_componentslab(f, leaves, ::Val{K}) where {K} = _callwith.(Ref(Component{K}(f)), leaves...)

_componentslabs(f, leaves, ::Val{0}) = ()
_componentslabs(f, leaves, ::Val{K}) where {K} = (_componentslabs(f, leaves, Val(K - 1))..., _componentslab(f, leaves, Val(K)))

# Component slabs are joined by nested `cat` along the field dims, innermost first, so the
# new storage is never written through a view.
function _assemble(slabs::Tuple, ::Val{S}, ::Val{N}) where {S, N}
    return only(_catlevels(slabs, Val(S), Val(N + 1)))
end

_catlevels(groups::Tuple, ::Val{()}, ::Val{D}) where {D} = groups
function _catlevels(groups::Tuple, ::Val{S}, ::Val{D}) where {S, D}
    return _catlevels(_catlevel(groups, Val(first(S)), Val(D)), Val(Base.tail(S)), Val(D + 1))
end

function _catlevel(groups::Tuple, ::Val{M}, ::Val{D}) where {M, D}
    return ntuple(j -> cat(ntuple(i -> groups[(j - 1) * M + i], Val(M))...; dims = Val(D)), Val(length(groups) ÷ M))
end

function _componentstorage(::Type{E}, shape::Val, bc::Broadcast.Broadcasted) where {E}
    g, leaves = _componentwise(bc)
    slabs = _componentslabs(g, leaves, Val(ncomponents(E)))
    return _assemble(slabs, shape, Val(length(axes(bc))))
end

function _componentwise_copy(bc::Broadcast.Broadcasted)
    E = Broadcast.combine_eltypes(bc.f, bc.args)
    if isfieldelement(E)
        return FieldDimArray{E, length(axes(bc))}(_componentstorage(E, Val(fieldshape(E)), bc))
    else
        g, leaves = _componentwise(bc)
        return _callwith.(Ref(g), leaves...)
    end
end

# All components are computed before `dest` is written, so `dest` may alias an argument.
function _componentwise_copyto!(dest::FieldDimArray{T}, bc::Broadcast.Broadcasted) where {T}
    axes(dest) == axes(bc) || throw(DimensionMismatch("destination axes $(axes(dest)) do not match broadcast axes $(axes(bc))"))
    parent(dest) .= _componentstorage(T, Val(_fielddims(dest)), bc)
    return dest
end

function _componentwise_copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted)
    g, leaves = _componentwise(bc)
    dest .= _callwith.(Ref(g), leaves...)
    return dest
end
