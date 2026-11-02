# FieldDimArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/FieldDimArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/FieldDimArrays.jl/dev/)
[![Build Status](https://github.com/ptiede/FieldDimArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/FieldDimArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/FieldDimArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/FieldDimArrays.jl)

`FieldDimArray{T}(parent)` presents a dense array `parent` as an array of elements of type
`T`. The leading dims of `parent` index the elements and its trailing dims hold their
components, so the storage stays one ordinary array (an `Array`, a GPU array, or a traced
Reactant array) while code reads and writes whole elements.

Element types are structs whose fields all have type `eltype(parent)` and, with StaticArrays
loaded, static arrays such as `SVector`, `SMatrix` and `FieldVector` subtypes. A 2×2 `SMatrix`
element uses two trailing dims of size 2 in column-major order.

```julia
using FieldDimArrays, StaticArrays

struct Point{T}
    x::T
    y::T
end

data = rand(100, 2)
a = FieldDimArray{Point}(data)     # 100-element array of Point{Float64}
a[3]                               # Point(data[3, 1], data[3, 2])
a.x                                # view(data, :, 1)
b = (p -> Point(p.y, -p.x)).(a)    # new FieldDimArray over a new 100×2 array
parent(b)                          # the dense storage

m = FieldDimArray{SMatrix{2, 2}}(rand(100, 2, 2))
m .* m .* adjoint.(m)              # FieldDimArray over a new 100×2×2 array
```

Broadcasts on the CPU make one pass over the elements. Broadcasts over traced Reactant arrays
run one broadcast per component over the component slabs and join the results with `cat`, so
the compiled program contains no loop over elements and keeps sharded dims split.

With AbstractFFTs loaded, `fft`, `bfft`, `ifft`, their in-place forms and their plans transform
the dense parent over the element dims, so every component is transformed in one call; the
region may only name element dims. `circshift!`, and with it `fftshift` and `ifftshift`, shift
the element dims of the parent.
