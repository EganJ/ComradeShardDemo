export fieldplot, fieldplot!, caltableplot

"""
    fieldplot(t, [bl], fieldx, fieldy; figure = (;), axis = (;), legend = true, kwargs...)
    fieldplot(pos, t, [bl], fieldx, fieldy; axis = (;), legend = true, kwargs...)
    fieldplot!(ax, t, [bl], fieldx, fieldy; kwargs...)

A scatter plot of the [`datafield`](@ref) `fieldy` against `fieldx` over the rows of the
[`VisTable`](@ref) or [`ClosureTable`](@ref) `t`, or over those on the baseline `bl`, e.g.
`(:AA, :AX)`. Each field is a name, such as `:uvdist`, a [`DataField`](@ref) or a function
of a row of [`datarows`](@ref). A complex `fieldy` plots its real part in `color` and its
imaginary part in `colorim`; a field with noise, such as `:measwnoise`, adds error bars.

The rows are split into series by the field `groupby` (default `:Fr`), or not split for
`nothing`. Unless `color` is given, each series takes the next color of the axis palette,
continuing after the series of earlier field plots in the axis. With `conjugate`, a
`:U`–`:V` plot also shows each point's conjugate.

`fieldplot` makes a figure, or with a grid position `pos` an axis, labeled by the fields,
with a legend entry per series, or a single one for a plot given a `label`. `figure` and
`axis` are passed to `Figure` and `Axis`, and `legend` to `axislegend` (`false` for none). A
field whose values are matrices, such as the measurements of a table of all correlations
(`Correlations()`), gets one axis per correlation on a 2×2 grid; the figure, or the matrix
of `AxisPlot`s, is returned. `fieldplot!` draws into the existing axis `ax` and leaves its
labels and legend alone; it also takes rows of [`datarows`](@ref) in place of `t`.

The other keywords are attributes of the plot. Needs a Makie backend.

# Examples
```julia-repl
julia> fig, ax, p = fieldplot(log_closure_amplitudes(t), :uvdist, :measwnoise)

julia> fieldplot!(ax, model, :uvdist, :measurement; color = :black, label = "Model")
```
"""
function fieldplot end
function fieldplot! end

"""
    caltableplot(gt...; width = 150, height = 125, layout = nothing, markers = nothing,
                 labels = nothing, figure = (;), axis = (;), legend = (;), kwargs...)

A figure of the calibration tables `gt`, with one axis per station, `width` × `height` in
size, laid out on a grid of `layout = (rows, columns)` (default 4 columns). Table `k` is
labeled `labels[k]`; frequency `j` uses the marker `markers[j]`. A `SiteArray` or
hierarchical instrument sample is first converted with [`caltable`](@ref).

`figure`, `axis` and `legend` are passed to `Figure`, each station's `Axis` and the `Legend`;
`axis.ylabel` labels only the bottom-left axis. The other keywords are passed to `scatter!`.
Needs a Makie backend.
"""
function caltableplot end
