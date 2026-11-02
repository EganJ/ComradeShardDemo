"""
    FieldDimArray{T}(parent)
    FieldDimArray{T, N}(parent)

An `AbstractArray{T, N}` whose elements are read from and written to the dense array
`parent`: the leading `N` dims of `parent` index the elements and its trailing dims index
their components, in column-major order. Every component of `T` must have type
`eltype(parent)`; a `UnionAll` `T` is completed with that type (`Point` over a `Float64`
parent becomes `Point{Float64}`).

The trailing dims of `parent` are either `fieldshape(T)` (one dim of length `ncomponents(T)`
for a struct, `(2, 2)` for a 2×2 static matrix) or a single dim of length `ncomponents(T)`.
`FieldDimArray{T}(parent)` uses `fieldshape(T)`; giving `N` selects the layout from
`ndims(parent) - N`.

`parent(a)` returns the dense storage, [`fieldview`](@ref) and `getproperty` return the
slab of one component as a view.

# Examples
```jldoctest
julia> struct Point{T}
           x::T
           y::T
       end

julia> a = FieldDimArray{Point}([1.0 3.0; 2.0 4.0]);

julia> a[2]
Point{Float64}(2.0, 4.0)

julia> a.y
2-element view(::Matrix{Float64}, :, 2) with eltype Float64:
 3.0
 4.0
```
"""
struct FieldDimArray{T, N, A <: AbstractArray} <: AbstractArray{T, N}
    parent::A
    function FieldDimArray{T, N, A}(parent) where {T, N, A}
        _checkparent(T, Val(N), parent)
        return new{T, N, A}(parent)
    end
end

function FieldDimArray{T, N}(parent::AbstractArray) where {T, N}
    T2 = _concrete(T, eltype(parent))
    return FieldDimArray{T2, N, typeof(parent)}(parent)
end

function FieldDimArray{T}(parent::AbstractArray) where {T}
    T2 = _concrete(T, eltype(parent))
    L = length(fieldshape(T2))
    ndims(parent) >= L ||
        throw(DimensionMismatch("a parent of $T2 elements needs at least $L dims for the components (trailing size $(fieldshape(T2))); got $(ndims(parent))"))
    return FieldDimArray{T2, ndims(parent) - L}(parent)
end

function _checkparent(::Type{T}, ::Val{N}, parent) where {T, N}
    (N isa Int && N >= 0) || throw(ArgumentError("the number of dims must be a nonnegative Int; got $N"))
    isconcretetype(T) || throw(ArgumentError("element type $T must be concrete"))
    S = componenttype(T)
    eltype(parent) === S ||
        throw(ArgumentError("the components of $T have type $S, but the parent has element type $(eltype(parent))"))
    shape = fieldshape(T)
    L = ndims(parent) - N
    (L == length(shape) || L == 1) ||
        throw(DimensionMismatch("a FieldDimArray{$T, $N} needs a parent with $(N + length(shape)) dims (trailing size $shape) or $(N + 1) dims (trailing size ($(ncomponents(T)),)); got $(ndims(parent)) dims"))
    trailing = ntuple(i -> size(parent, N + i), Val(L))
    expected = _layoutshape(T, Val(L))
    trailing == expected ||
        throw(DimensionMismatch("the trailing size of the parent must be $expected to hold the components of $T; got $trailing"))
    Base.require_one_based_indexing(parent)
    return nothing
end

_layoutshape(::Type{T}, ::Val{L}) where {T, L} = L == length(fieldshape(T)) ? fieldshape(T) : (ncomponents(T),)

_nfielddims(::Type{<:FieldDimArray{T, N, A}}) where {T, N, A} = ndims(A) - N
_fielddims(::Type{V}) where {V <: FieldDimArray} = _layoutshape(eltype(V), Val(_nfielddims(V)))
_fielddims(a::FieldDimArray) = _fielddims(typeof(a))

_fieldindex(a::FieldDimArray, k::Int) = Tuple(CartesianIndices(_fielddims(a))[k])
_colons(::Val{N}) where {N} = ntuple(_ -> Colon(), Val(N))

Base.parent(a::FieldDimArray) = getfield(a, :parent)

Base.size(a::FieldDimArray{T, N}) where {T, N} = ntuple(d -> size(parent(a), d), Val(N))
Base.axes(a::FieldDimArray{T, N}) where {T, N} = ntuple(d -> axes(parent(a), d), Val(N))
Base.IndexStyle(::Type{<:FieldDimArray{T, N, A}}) where {T, N, A} = IndexStyle(A)

function _checkcartesian(a::FieldDimArray{T}, I) where {T}
    checkbounds(a, I...)
    checkbounds(parent(a), I..., _fieldindex(a, ncomponents(T))...)
    return nothing
end

function _checklinear(a::FieldDimArray{T}, i) where {T}
    checkbounds(a, i)
    checkbounds(parent(a), i + (ncomponents(T) - 1) * length(a))
    return nothing
end

# `@inbounds` does not reach into the `ntuple` closures; `@boundscheck` covers each access.
Base.@propagate_inbounds function Base.getindex(a::FieldDimArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck _checkcartesian(a, I)
    p = parent(a)
    return fromcomponents(T, ntuple(k -> @inbounds(p[I..., _fieldindex(a, k)...]), Val(ncomponents(T))))
end

Base.@propagate_inbounds function Base.getindex(a::FieldDimArray{T}, i::Int) where {T}
    @boundscheck _checklinear(a, i)
    p = parent(a)
    n = length(a)
    return fromcomponents(T, ntuple(k -> @inbounds(p[i + (k - 1) * n]), Val(ncomponents(T))))
end

Base.@propagate_inbounds function Base.setindex!(a::FieldDimArray{T, N}, v, I::Vararg{Int, N}) where {T, N}
    @boundscheck _checkcartesian(a, I)
    p = parent(a)
    c = components(convert(T, v))
    ntuple(k -> @inbounds(p[I..., _fieldindex(a, k)...] = c[k]), Val(ncomponents(T)))
    return a
end

Base.@propagate_inbounds function Base.setindex!(a::FieldDimArray{T}, v, i::Int) where {T}
    @boundscheck _checklinear(a, i)
    p = parent(a)
    n = length(a)
    c = components(convert(T, v))
    ntuple(k -> @inbounds(p[i + (k - 1) * n] = c[k]), Val(ncomponents(T)))
    return a
end

const LeadingIndex = Union{Integer, Colon, AbstractVector{<:Integer}}

Base.@propagate_inbounds function Base.view(a::FieldDimArray{T, N}, I::Vararg{LeadingIndex, N}) where {T, N}
    J = to_indices(a, I)
    @boundscheck checkbounds(a, J...)
    return _rewrap(a, view(parent(a), J..., _colons(Val(_nfielddims(typeof(a))))...))
end

Base.@propagate_inbounds function Base.getindex(a::FieldDimArray{T, N}, I::Vararg{LeadingIndex, N}) where {T, N}
    J = to_indices(a, I)
    @boundscheck checkbounds(a, J...)
    return _leadinggetindex(a, J)
end

_leadinggetindex(a, J::Tuple{Vararg{Int}}) = a[J...]
_leadinggetindex(a, J) = _rewrap(a, parent(a)[J..., _colons(Val(_nfielddims(typeof(a))))...])

function _rewrap(a::FieldDimArray{T}, p::AbstractArray) where {T}
    return FieldDimArray{_withcomponents(T, eltype(p)), ndims(p) - _nfielddims(typeof(a))}(p)
end

"""
    fieldview(a::FieldDimArray, k::Integer)
    fieldview(a::FieldDimArray, i::Integer, j::Integer, ...)
    fieldview(a::FieldDimArray, name::Symbol)

The values of one component of every element of `a`, as a view of `parent(a)`. The component
is given by its position `k` in column-major order, by its indices into the trailing dims of
`parent(a)`, or by its name (see `propertynames(a)`); `a.name` is `fieldview(a, :name)`.
"""
function fieldview(a::FieldDimArray{T, N}, k::Integer) where {T, N}
    n = ncomponents(T)
    1 <= k <= n || throw(ArgumentError("component index $k is out of range 1:$n for $T"))
    return view(parent(a), _colons(Val(N))..., _fieldindex(a, Int(k))...)
end

function fieldview(a::FieldDimArray{T, N}, i::Integer, j::Integer, I::Integer...) where {T, N}
    shape = _fielddims(a)
    idx = (i, j, I...)
    length(idx) == length(shape) && all(map((x, s) -> 1 <= x <= s, idx, shape)) ||
        throw(ArgumentError("component indices $idx do not index the trailing size $shape of $T components"))
    return view(parent(a), _colons(Val(N))..., idx...)
end

fieldview(a::FieldDimArray{T}, name::Symbol) where {T} = fieldview(a, _componentindex(T, name))

function _componentindex(::Type{T}, name::Symbol) where {T}
    k = findfirst(==(name), componentnames(T))
    k === nothing && throw(ArgumentError("$T has no component named $(repr(name)); component names are $(componentnames(T))"))
    return k
end

Base.getproperty(a::FieldDimArray, name::Symbol) = fieldview(a, name)
Base.propertynames(a::FieldDimArray) = componentnames(eltype(a))

"""
    elementview(::Type{T}, parent::AbstractArray)
    elementview(a::AbstractArray)

The array whose elements are the logical elements of a container. `elementview(T, parent)` is
`FieldDimArray{T}(parent)`, or `parent` itself when `eltype(parent) <: T`. An array is its own
element view. Containers that wrap dense storage add a method returning the array of their
elements.
"""
elementview(::Type{T}, parent::AbstractArray) where {T} = eltype(parent) <: T ? parent : FieldDimArray{T}(parent)
elementview(a::AbstractArray) = a

function Base.similar(a::FieldDimArray{T}, ::Type{S}, dims::Dims) where {T, S}
    p = parent(a)
    if S === T
        return FieldDimArray{T, length(dims)}(similar(p, (dims..., _fielddims(a)...)))
    elseif isfieldelement(S)
        return FieldDimArray{S, length(dims)}(similar(p, componenttype(S), (dims..., fieldshape(S)...)))
    else
        return similar(p, S, dims)
    end
end

Base.copy(a::FieldDimArray{T, N}) where {T, N} = FieldDimArray{T, N}(copy(parent(a)))

function Base.circshift!(dest::FieldDimArray{T, N}, src::FieldDimArray, shiftamt::Base.DimsInteger) where {T, N}
    length(shiftamt) <= N ||
        throw(ArgumentError("$(length(shiftamt)) shifts given for a FieldDimArray with $N dims"))
    circshift!(parent(dest), parent(src), (Base.fill_to_length(shiftamt, 0, Val(N))..., map(zero, _fielddims(dest))...))
    return dest
end

Base.circshift!(dest::FieldDimArray, src::FieldDimArray, ::Tuple{}) = (copyto!(parent(dest), parent(src)); dest)

Base.dataids(a::FieldDimArray) = Base.dataids(parent(a))
Base.unaliascopy(a::FieldDimArray{T, N}) where {T, N} = FieldDimArray{T, N}(Base.unaliascopy(parent(a)))

function Base.showarg(io::IO, a::FieldDimArray{T}, toplevel) where {T}
    print(io, "FieldDimArray{", T, "}(")
    Base.showarg(io, parent(a), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", T)
    return nothing
end
