module FieldDimArraysReactantExt

using Reactant: Reactant, AnyTracedRArray
using Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle
using FieldDimArrays: FieldDimArrays, FieldDimArray, FieldDimArrayStyle, _withcomponents

FieldDimArrays.componentwise(::AnyTracedRArray) = true

function Base.BroadcastStyle(::FieldDimArrayStyle{N}, ::AbstractReactantArrayStyle{M}) where {N, M}
    return FieldDimArrayStyle{max(N, M)}()
end

# The element type of a traced `FieldDimArray` follows the element type of its traced parent.
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(V::Type{<:FieldDimArray}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    V isa DataType || return V
    T, N, A = V.parameters
    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, ndevices, runtime)
    return FieldDimArray{_withcomponents(T, eltype(A2)), N, A2}
end

end
