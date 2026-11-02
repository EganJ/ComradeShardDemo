module FieldDimArraysStaticArraysExt

using StaticArrays: StaticArray, FieldArray, similar_type
import FieldDimArrays: ncomponents, fieldshape, componenttype, componentnames, fromcomponents,
    components, withcomponenttype, isfieldelement

ncomponents(::Type{T}) where {T <: StaticArray} = length(T)
fieldshape(::Type{T}) where {T <: StaticArray} = size(T)
componenttype(::Type{T}) where {T <: StaticArray} = eltype(T)
componentnames(::Type{<:StaticArray}) = ()
componentnames(::Type{T}) where {T <: FieldArray} = fieldnames(T)
fromcomponents(::Type{T}, c::Tuple) where {T <: StaticArray} = T(c)
components(x::StaticArray) = Tuple(x)
isfieldelement(::Type{T}) where {T <: StaticArray} = isconcretetype(T) && !ismutabletype(T)

function withcomponenttype(::Type{T}, ::Type{S}) where {T <: StaticArray, S}
    isconcretetype(T) && eltype(T) === S && return T
    T2 = similar_type(T, S)
    (isconcretetype(T2) && eltype(T2) === S) ||
        throw(ArgumentError("cannot build a static array type like $T with elements of type $S; got $T2"))
    return T2
end

# `similar_type` of a field array is an `SArray`; a field array whose element type is its only
# type parameter keeps its own type.
function withcomponenttype(::Type{T}, ::Type{S}) where {T <: FieldArray, S}
    isconcretetype(T) && eltype(T) === S && return T
    W = Base.typename(T).wrapper
    T2 = W isa UnionAll ? W{S} : W
    (isconcretetype(T2) && eltype(T2) === S) ||
        throw(ArgumentError("cannot build a field array type like $T with elements of type $S; define `FieldDimArrays.withcomponenttype` for it"))
    return T2
end

end
