module VLBIInstrumentModelsMakieExt

using Makie
using VLBIInstrumentModels
using VLBIInstrumentModels: DataField, datafield, _rowU, _rowV
using Printf
import Measurements

import VLBIInstrumentModels: fieldplot, fieldplot!, caltableplot

const MakieGrid = Union{Makie.GridPosition, Makie.GridSubposition}

# The tables are plotted through their rows, `datarows(t)`.
const _Table = Union{VLBIInstrumentModels.VisTable, VLBIInstrumentModels.ClosureTable}
const _Baseline = Union{Colon, NTuple{2, Union{String, Symbol}}}

Makie.@recipe FieldPlot (data::AbstractVector, bl::Any, fieldx::Any, fieldy::Any) begin
    "The field whose values split the rows into series; `nothing` for one series"
    groupby = :Fr
    "Whether a `:U`–`:V` plot also shows each point's conjugate"
    conjugate = true
    "The color of every series; `automatic` continues the axis palette, one color per series"
    color = Makie.automatic
    "The color of the imaginary part of a complex `fieldy`"
    colorim = @inherit linecolor
    "The marker type"
    marker = @inherit marker
    "The size of the markers"
    markersize = @inherit markersize
    "The color around the markers"
    strokecolor = @inherit markerstrokecolor
    "The width of the color around the markers"
    strokewidth = @inherit markerstrokewidth
    "The width of the error bars"
    linewidth = @inherit linewidth
    "The cap of the error bars"
    linecap = @inherit linecap
    "The width of the error bar caps"
    linecapwidth = 0
    Makie.mixin_generic_plot_attributes()...
    Makie.mixin_colormap_attributes()...
end

Makie.convert_arguments(::Type{<:FieldPlot}, rows::AbstractVector, fieldx, fieldy) = (rows, Colon(), fieldx, fieldy)
Makie.convert_arguments(::Type{<:FieldPlot}, rows::AbstractVector, bl, fieldx, fieldy) = (rows, bl, fieldx, fieldy)
Makie.convert_arguments(P::Type{<:FieldPlot}, t::_Table, args...) = Makie.convert_arguments(P, datarows(t), args...)

# The rows on the baseline `bl`, in either order.
_onbaseline(rows, ::Colon) = rows
function _onbaseline(rows, bl)
    hasproperty(rows, :sites) || throw(ArgumentError("closures are not on one baseline"))
    s = Symbol.(bl)
    on = filter(r -> r.sites == s || reverse(r.sites) == s, rows)
    isempty(on) && throw(ArgumentError("no data on the baseline $bl"))
    return on
end

_groups(rows, ::Nothing) = [nothing => rows]
function _groups(rows, groupby)
    vals = map(datafield(groupby), rows)
    return [k => rows[isequal.(vals, Ref(k))] for k in unique(vals)]
end

_isuv(fx::DataField, fy::DataField) = fx.value in (_rowU, _rowV) && fy.value in (_rowU, _rowV)

# The x and y values of each group of `rows`, with the conjugate points of a `:U`–`:V` plot.
function _series(rows, fx, fy, groupby, conjugate)
    withconj = conjugate && _isuv(fx, fy)
    return map(_groups(rows, groupby)) do (key, grp)
        x, y = _value.(fx.(grp)), fy.(grp)
        withconj && return (; key, x = vcat(x, -x), y = vcat(y, -y))
        return (; key, x, y)
    end
end

function _group(gs, i, n)
    length(gs) == n || throw(ArgumentError("the data now splits into $(length(gs)) groups instead of $n; plot it anew"))
    return gs[i]
end

_value(v::Measurements.Measurement) = Measurements.value(v)
_value(v) = v
_uncertainty(v::Measurements.Measurement) = Measurements.uncertainty(v)

const _WithNoise = Union{Measurements.Measurement, Complex{<:Measurements.Measurement}}

# The automatic colors of the `n` series of `p`: the palette of its axis, continued after the
# series of the automatically colored field plots drawn there before `p`.
function _autocolors(p, n)
    scene = Makie.parent_scene(p)
    palette = scene.theme.palette.color[]
    before = Iterators.takewhile(q -> q !== p, scene.plots)
    offset = sum(q -> q isa FieldPlot && q.color[] === Makie.automatic ? length(q.groups[]) : 0, before; init = 0)
    return [palette[mod1(offset + i, length(palette))] for i in 1:n]
end

function Makie.plot!(p::FieldPlot{<:Tuple{<:AbstractVector, <:_Baseline, <:Any, <:Any}})
    map!(p, [:data, :bl, :fieldx, :fieldy, :groupby, :conjugate], :groups) do data, bl, fieldx, fieldy, groupby, conjugate
        return _series(_onbaseline(data, bl), datafield(fieldx), datafield(fieldy), groupby, conjugate)
    end
    groups = p.groups[]
    y = first(groups).y
    eltype(y) <: AbstractMatrix && throw(
        ArgumentError("the values of `fieldy` are matrices; plot one entry, e.g. `r -> r.measurement[1, 1]`")
    )
    parts = eltype(y) <: Complex ? ((real, false), (imag, true)) : ((identity, false),)
    auto = _autocolors(p, length(groups))
    for i in eachindex(groups)
        g = lift(gs -> _group(gs, i, length(groups)), p.groups)
        x = lift(g -> g.x, g)
        color = lift(c -> c === Makie.automatic ? auto[i] : c, p.color)
        for (part, isim) in parts
            c = isim ? p.colorim : color
            v = lift(g -> _value.(part.(g.y)), g)
            if eltype(y) <: _WithNoise
                e = lift(g -> _uncertainty.(part.(g.y)), g)
                errorbars!(
                    p, p.attributes, x, v, e;
                    color = c, linewidth = p.linewidth, linecap = p.linecap, whiskerwidth = p.linecapwidth,
                )
            end
            scatter!(
                p, p.attributes, x, v;
                color = c, marker = p.marker, markersize = p.markersize,
                strokecolor = p.strokecolor, strokewidth = p.strokewidth,
            )
        end
    end
    return p
end

function frequencylabel(ν::Number)
    ν >= 1.0e12 && return @sprintf("%.2f THz", ν / 1.0e12)
    ν >= 1.0e9 && return @sprintf("%.2f GHz", ν / 1.0e9)
    ν >= 1.0e6 && return @sprintf("%.2f MHz", ν / 1.0e6)
    ν >= 1.0e3 && return @sprintf("%.2f kHz", ν / 1.0e3)
    return @sprintf("%.2f Hz", ν)
end

measname(t::VLBIInstrumentModels.VisTable) = _measname(eltype(VLBIInstrumentModels.measurement(t)))
_measname(::Type{<:Complex}) = "Visibility (Jy)"
_measname(::Type{<:Real}) = "Visibility Amplitude (Jy)"
_measname(::Type{<:AbstractMatrix}) = "Correlation (Jy)"
measname(ct::VLBIInstrumentModels.ClosureTable) = ct.product isa VLBIInstrumentModels.ClosurePhases ? "Closure Phase (rad)" : "Log Closure Amplitude"

function fieldplot(t::_Table, args...; figure = (;), kwargs...)
    fig = Figure(; figure...)
    return _figureplot(fig, fieldplot(fig[1, 1], t, args...; kwargs...))
end

_figureplot(fig, ap::Makie.AxisPlot) = Makie.FigureAxisPlot(fig, ap.axis, ap.plot)
_figureplot(fig, ::AbstractMatrix) = fig

fieldplot(pos::MakieGrid, t::_Table, fieldx, fieldy; kwargs...) = fieldplot(pos, t, Colon(), fieldx, fieldy; kwargs...)
function fieldplot(pos::MakieGrid, t::_Table, bl::_Baseline, fieldx, fieldy; axis = (;), kwargs...)
    bl isa Colon || (axis = merge((; title = string(bl[1], " - ", bl[2])), axis))
    rows = _onbaseline(datarows(t), bl)
    fx, fy = datafield(fieldx), datafield(fieldy)
    if !isempty(rows) && fy(first(rows)) isa AbstractMatrix
        g = GridLayout(pos)
        return [_axisplot(g[i, j], rows, measname(t), fx, _entry(fy, i, j); axis, kwargs...) for i in 1:2, j in 1:2]
    end
    return _axisplot(pos, rows, measname(t), fx, fy; axis, kwargs...)
end

# The entry `(i, j)` of a field whose values are matrices.
function _entry(f::DataField, i, j)
    label = isnothing(f.label) ? nothing : string(f.label, " ", (i, j))
    return DataField(r -> f(r)[i, j], label, f.title)
end

_label(f::DataField, mname) = something(f.label, mname)

_grouplabel(groupby, key) = groupby in (:Fr, Fr) ? frequencylabel(key) : _keylabel(key)
_keylabel(k::Tuple) = join(k, "-")
_keylabel(k) = string(k)

# A new axis at `pos` with the field plot of `fy` against `fx` over `rows`, whose
# measurements are named `mname`.
function _axisplot(pos, rows, mname, fx, fy; axis, legend = true, conjugate = true, kwargs...)
    defaults = (; xlabel = _label(fx, mname), ylabel = _label(fy, mname))
    isnothing(fy.title) || (defaults = merge(defaults, (; title = fy.title(fy.(rows)))))
    conjugate && _isuv(fx, fy) && (defaults = merge(defaults, (; aspect = 1)))
    fx.value === _rowU && (defaults = merge(defaults, (; xreversed = true)))
    ax = Axis(pos; merge(defaults, axis)...)
    fp = fieldplot!(ax, rows, fx, fy; conjugate, kwargs...)
    legend === false || _legend!(ax, fp, haskey(kwargs, :label), legend === true ? (;) : legend)
    return Makie.AxisPlot(ax, fp)
end

# A plot given a `label` gets one legend entry; otherwise each series gets one.
function _legend!(ax, fp, labeled, opts)
    labeled && return axislegend(ax; merge = true, unique = true, opts...)
    groupby = fp.groupby[]
    isnothing(groupby) && return nothing
    keys = map(g -> g.key, fp.groups[])
    colors = fp.color[] === Makie.automatic ? _autocolors(fp, length(keys)) : fill(fp.color[], length(keys))
    entries = [MarkerElement(; color, marker = fp.marker[], markersize = fp.markersize[]) for color in colors]
    return axislegend(ax, entries, map(k -> _grouplabel(groupby, k), keys); opts...)
end

function caltableplot(
        tables...;
        width = 150,
        height = 125,
        layout = nothing,
        markers = nothing,
        labels = nothing,
        figure = (;),
        axis = (;),
        legend = (;),
        kwargs...,
    )
    gt = map(prepare_caltable, tables)
    sitelist = sites(argmax(x -> length(sites(x)), gt))
    tup = maximum(maximum.(x -> x[:Ti].t0 + x[:Ti].dt / 2, gt))
    tlo = minimum(minimum.(x -> x[:Ti].t0 - x[:Ti].dt / 2, gt))

    lims = get(axis, :limits, ((tlo, tup), nothing))
    lims = isnothing(lims[1]) ? ((tlo, tup), lims[2]) : lims
    ylabel = get(axis, :ylabel, nothing)
    axis = Base.structdiff(axis, NamedTuple{(:limits, :ylabel)})

    rowlen, collen = isnothing(layout) ? (ceil(Int, length(sitelist) / 4), 4) : layout

    # The extra 50 px hold the time label.
    fig = Figure(; size = (width * collen, height * rowlen + 50), figure...)
    cells = Iterators.filter(c -> collen * (c[1] - 1) + c[2] <= length(sitelist), Iterators.product(1:rowlen, 1:collen))
    axs = map(cells) do (i, j)
        site = sitelist[collen * (i - 1) + j]
        site => Axis(fig[i, j]; title = string(site), width, height, limits = lims, axis...)
    end

    markers = isnothing(markers) ? collect(keys(Makie.default_marker_map())) : markers
    labels = isnothing(labels) ? fill(" ", length(gt)) : labels

    for (site, ax) in axs, (k, gi) in pairs(gt)
        for (j, ν) in pairs(unique(gi.Fr))
            at = findall(==(ν), gi.Fr)
            x = getproperty.(gi.Ti, :t0)[at]
            y = getproperty(gi, site)[at]
            label = string(labels[k], " ", frequencylabel(round(ν.central, digits = 2)))
            sc = scatter!(ax, x, map(_calvalue, y); marker = markers[j], label, kwargs...)
            any(v -> v isa Measurements.Measurement, y) && errorbars!(ax, x, map(_calvalue, y), map(_caluncertainty, y); color = sc.color)
        end
    end

    Legend(fig[1, collen + 1], last(axs[end]); legend...)
    last(axs[rowlen]).xlabel = "Time (UTC)"
    isnothing(ylabel) || (last(axs[rowlen]).ylabel = ylabel)
    resize_to_layout!(fig)
    return fig
end

_calvalue(::Missing) = NaN
_calvalue(v) = _value(v)
_caluncertainty(::Missing) = NaN
_caluncertainty(v::Measurements.Measurement) = Measurements.uncertainty(v)
_caluncertainty(v) = zero(v)

prepare_caltable(gt::VLBIInstrumentModels.CalTable) = gt
prepare_caltable(gt::Union{VLBIInstrumentModels.SiteArray, VLBIInstrumentModels.HierarchicalSiteSample}) = caltable(gt)

end
