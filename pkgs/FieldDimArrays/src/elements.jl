"""
    ncomponents(::Type{T}) -> Int

The number of components of an element of type `T`. For a struct this is its number of fields.
"""
ncomponents(::Type{T}) where {T} = fieldcount(T)

"""
    fieldshape(::Type{T}) -> NTuple{L, Int}

The sizes of the trailing parent dims that hold the components of an element of type `T`
by default, in column-major component order. Their product is `ncomponents(T)`. For a
struct this is `(ncomponents(T),)`; for a static array it is its size.
"""
fieldshape(::Type{T}) where {T} = (ncomponents(T),)

"""
    componenttype(::Type{T}) -> Type

The type shared by every component of an element of type `T`. For a struct this is the
type of its fields, which must all be the same.
"""
function componenttype(::Type{T}) where {T}
    fieldcount(T) > 0 || throw(ArgumentError("$T has no fields to use as components"))
    _ishomogeneous(T) || throw(ArgumentError("the fields of $T must all have the same type; found $(fieldtypes(T))"))
    return fieldtype(T, 1)
end

_ishomogeneous(::Type{T}) where {T} = all(==(fieldtype(T, 1)), fieldtypes(T))

"""
    componentnames(::Type{T}) -> Tuple{Vararg{Symbol}}

The names under which the components of `T` are accessible with `getproperty` on a
`FieldDimArray{T}`. For a struct these are its field names.
"""
componentnames(::Type{T}) where {T} = fieldnames(T)

"""
    fromcomponents(::Type{T}, c::NTuple) -> T

Build an element of type `T` from its components in order. For a struct this calls
`T(c...)`.
"""
fromcomponents(::Type{T}, c::Tuple) where {T} = T(c...)
fromcomponents(::Type{T}, c::Tuple) where {T <: Tuple} = T(c)
fromcomponents(::Type{T}, c::Tuple) where {T <: NamedTuple} = T(c)

"""
    components(x) -> NTuple

The components of `x` in order, the inverse of [`fromcomponents`](@ref). For a struct these
are its fields.
"""
components(x) = ntuple(i -> getfield(x, i), Val(nfields(x)))
components(x::Tuple) = x
components(x::NamedTuple) = Tuple(x)

"""
    withcomponenttype(::Type{T}, ::Type{S}) -> Type

The concrete element type of the same kind as `T` whose components have type `S`. `T` may be
a `UnionAll`, as in `withcomponenttype(Point, Float64) == Point{Float64}`. For a struct this is
`W{S}`, where `W` is the type `T` without its parameters, and it throws an `ArgumentError` when
that type does not have components of type `S`.
"""
function withcomponenttype(::Type{T}, ::Type{S}) where {T, S}
    isconcretetype(T) && componenttype(T) === S && return T
    W = Base.typename(T).wrapper
    W isa UnionAll || throw(ArgumentError("cannot build an element type like $T with components of type $S: $W has no type parameters"))
    T2 = W{S}
    (isconcretetype(T2) && componenttype(T2) === S) ||
        throw(ArgumentError("cannot build an element type like $T with components of type $S: $T2 is not a concrete type with components of type $S; define `FieldDimArrays.withcomponenttype` for it"))
    return T2
end

"""
    isfieldelement(::Type{T}) -> Bool

Whether results of type `T` produced by broadcasting over a `FieldDimArray` are stored as a
new `FieldDimArray{T}` over dense storage. True for concrete immutable structs whose fields
all have one type, except `Number`s, `Tuple`s and `NamedTuple`s, which are stored as plain
arrays.
"""
function isfieldelement(::Type{T}) where {T}
    return isconcretetype(T) && isstructtype(T) && !ismutabletype(T) &&
        !(T <: Union{Number, Tuple, NamedTuple}) && fieldcount(T) > 0 && _ishomogeneous(T)
end

_concrete(::Type{T}, ::Type{S}) where {T, S} = isconcretetype(T) ? T : withcomponenttype(T, S)

_withcomponents(::Type{T}, ::Type{S}) where {T, S} = componenttype(T) === S ? T : withcomponenttype(T, S)
