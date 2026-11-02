module FieldDimArraysAdaptExt

using Adapt: Adapt
using FieldDimArrays: FieldDimArray, _rewrap

Adapt.adapt_structure(to, a::FieldDimArray) = _rewrap(a, Adapt.adapt(to, parent(a)))
Adapt.parent_type(::Type{<:FieldDimArray{T, N, A}}) where {T, N, A} = A

end
