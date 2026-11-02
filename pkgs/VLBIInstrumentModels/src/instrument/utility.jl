export site_tuple

"""
    site_tuple(sites, default; kwargs...)
    site_tuple(t::VisTable, default; kwargs...)

Convienence function that will construct a `NamedTuple` of objects
whose names are the `sites` of the visibility table `t` or explicitly in the argument
`sites`. The `NamedTuple` will be filled with `default` if no kwargs are defined
otherwise each kwarg (key, value) pair denotes a sites and value pair.

## Examples
```julia-repl
julia> sites = (:AA, :AP, :LM, :PV)
julia> site_tuple(sites, ScanSeg())
(AA = ScanSeg(), AP = ScanSeg(), LM = ScanSeg(), PV = ScanSeg())
julia> site_tuple(sites, ScanSeg(); AA = FixedSeg(1.0))
(AA = FixedSeg(1.0), AP = ScanSeg(), LM = ScanSeg(), PV = ScanSeg())
julia> site_tuple(sites, ScanSeg(); AA = FixedSeg(1.0), PV = TrackSeg())
(AA = FixedSeg(1.0), AP = ScanSeg(), LM = ScanSeg(), PV = TrackSeg())
```
"""
function site_tuple(sites::NTuple{N, Symbol}, default; kwargs...) where {N}
    out = map(x -> get(kwargs, x, default), sites)
    return NamedTuple{sites}(out)
end
site_tuple(t::VisTable, default; kwargs...) = site_tuple(sites(t), default; kwargs...)
site_tuple(st::AbstractVector{Symbol}, default; kwargs...) = site_tuple(Tuple(st), default; kwargs...)
