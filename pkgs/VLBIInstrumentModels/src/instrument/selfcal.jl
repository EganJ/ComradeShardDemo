export selfcal_phase

# The instrument parameter of the residual phase about the self-calibration solution.
const SELFCAL_RESIDUAL = :selfcal_residual
# Power iterations on the station matrix: φ̂ is the phase of A^SELFCAL_ITERATIONS e_ref.
const SELFCAL_ITERATIONS = 32

struct ObservedSelfCalInstrumentModel{M <: ObservedInstrumentModel, W, G, L, K, E} <: AbstractInstrumentModel
    model::M
    phase::Symbol
    wdata::W
    weight::G
    layout::L
    iterations::Int
    # the measured values, by record then channel
    keep::K
    # the residual's entries at each value's first and second station
    entry1::E
    entry2::E
end

EnzymeRules.inactive_type(::Type{<:ObservedSelfCalInstrumentModel}) = true

function Base.show(io::IO, mime::MIME"text/plain", m::ObservedSelfCalInstrumentModel)
    printstyled(io, "ObservedSelfCalInstrumentModel"; bold = true, color = :light_cyan)
    println(io, " (station phase = self-calibration + $(m.phase))")
    return show(io, mime, m.model)
end

# `model` with the residual phase added, referenced by `refant`.
function _with_residual(model::InstrumentModel, refant)
    # Uniform on the circle to a relative 5e-9, with one latent coordinate per entry, no singular
    # point, and no seam at ±π.
    residual = ArrayPrior(IIDSitePrior(IntegSeg(), WrappedNormal(0.0, 2π)); refant)
    prior = merge(model.prior, NamedTuple{(SELFCAL_RESIDUAL,)}((residual,)))
    return _instrumentmodel(model.jones, prior, model.refbasis, model.gaugefix, false, model.feedrotation_corrected)
end

function _set_selfcal(m::InstrumentModel, t::VisTable)
    eltype(measurement(t)) <: Real && throw(
        ArgumentError("a self-calibrated instrument model is solved against visibilities or coherencies, got a table of amplitudes")
    )
    keep = _measuredvalues(t)
    wdata, weight = _selfcal_weights(t, keep)
    # The entries of the residual and the baselines between them do not depend on which are
    # fixed, so a first build without references gives the groups the references come from.
    intfree, priorfree = observe(_with_residual(m, NoReference()), t)
    smap = sitelookup(getproperty(priorfree, SELFCAL_RESIDUAL))
    e1, e2 = _entrymaps(intfree, SELFCAL_RESIDUAL)
    rowweight = sum(map(w -> collect(w), weight))
    layout = _selfcal_layout(smap, e1[keep], e2[keep], rowweight)
    refs = unique(layout.refentry)
    intobs, prior = observe(_with_residual(m, EntryReference(refs, 0.0)), t)
    sort(fixed_indices(getproperty(prior, SELFCAL_RESIDUAL))) == sort(refs) ||
        error("the residual phase prior did not fix exactly the reference entries")
    entry1, entry2 = _entrymaps(intobs, SELFCAL_RESIDUAL)
    return ObservedSelfCalInstrumentModel(
        intobs, SELFCAL_RESIDUAL, wdata, weight, layout, SELFCAL_ITERATIONS, keep, entry1, entry2
    ), prior
end

# The entries of the term `name` at each value's first and second station.
function _entrymaps(J::ObservedInstrumentModel, name)
    L = layout(J)
    e = getproperty(L.entries, name)
    return e[L.cell1], e[L.cell2]
end

# Index maps for the station matrices of every group, a set of entries of the residual at one
# time and frequency that baselines link. A group's reference is its entry with the largest
# summed weight `rowweight` over the baselines it is on. The groups with `n` stations form one
# stack of matrices `A[a, b, j]` and station vectors `x[a, j]`:
#   - `pick[a, b, j]` indexes `[0; x; conj(x)]` for a per-baseline `x`: `1 + k` when baseline
#     `k` runs from station `a` to `b` of the `j`-th group, `1 + K + k` when it runs from `b`
#     to `a`, and `1` when no baseline joins them.
#   - `start[a, j]` indexes `A[a, r, j]`, where `r` is the reference station of group `j`.
#   - `entry[i]` indexes the station vectors at the stack's `i`-th entry, and `refslot[i]` at
#     the reference entry of its group.
# `order` sorts the entries of the stacks, concatenated, into the order of the residual.
function _selfcal_layout(smap::SiteLookup, i1::AbstractVector{<:Integer}, i2::AbstractVector{<:Integer}, rowweight)
    st = sites(smap)
    stamps = collect(zip(times(smap), frequencies(smap)))
    describe(e) = "site = $(st[e]), t = $(_center(times(smap)[e]))"

    root = collect(eachindex(st))
    function findroot(e)
        while root[e] != e
            root[e] = root[root[e]]
            e = root[e]
        end
        return e
    end
    for k in eachindex(i1, i2)
        e1, e2 = i1[k], i2[k]
        stamps[e1] == stamps[e2] || throw(
            ArgumentError("baseline $k joins $(describe(e1)) and $(describe(e2)) at different times")
        )
        root[findroot(e1)] = findroot(e2)
    end
    roots = map(findroot, eachindex(st))
    group_of = Dict(r => g for (g, r) in pairs(unique(roots)))
    ge = [group_of[r] for r in roots]
    G = length(group_of)
    groupsites = [sort(unique(st[e] for e in eachindex(ge) if ge[e] == g)) for g in 1:G]
    ne = [findfirst(==(st[e]), groupsites[ge[e]]) for e in eachindex(st)]
    nstations = map(length, groupsites)

    entryweight = zeros(length(st))
    for k in eachindex(i1, i2, rowweight)
        entryweight[i1[k]] += rowweight[k]
        entryweight[i2[k]] += rowweight[k]
    end
    ref = zeros(Int, G)
    for e in eachindex(ge)
        g = ge[e]
        (ref[g] == 0 || entryweight[e] > entryweight[ref[g]]) && (ref[g] = e)
    end
    refentry = [ref[ge[e]] for e in eachindex(ge)]

    sizes = sort(unique(nstations))
    members = [findall(==(n), nstations) for n in sizes]
    stack = [searchsortedfirst(sizes, n) for n in nstations]
    slot = zeros(Int, G)
    for m in members
        slot[m] .= eachindex(m)
    end

    K = length(i1)
    picks = [ones(Int, n, n, length(m)) for (n, m) in zip(sizes, members)]
    for k in eachindex(i1, i2)
        e1, e2 = i1[k], i2[k]
        g, a, b = ge[e1], ne[e1], ne[e2]
        pick = picks[stack[g]]
        pick[a, b, slot[g]] == 1 ||
            throw(ArgumentError("baseline $k repeats the station pair of $(describe(e1)) and $(describe(e2))"))
        pick[a, b, slot[g]] = 1 + k
        pick[b, a, slot[g]] = 1 + K + k
    end

    entries = [findall(e -> stack[ge[e]] == j, eachindex(ge)) for j in eachindex(sizes)]
    stacks = map(eachindex(sizes)) do j
        n = sizes[j]
        LA, LX = LinearIndices(picks[j]), LinearIndices((n, length(members[j])))
        start = [LA[a, ne[ref[g]], slot[g]] for a in 1:n, g in members[j]]
        entry = [LX[ne[e], slot[ge[e]]] for e in entries[j]]
        refslot = [LX[ne[refentry[e]], slot[ge[e]]] for e in entries[j]]
        eye = reshape([a == b ? 1.0 : 0.0 for a in 1:n, b in 1:n], n, n, 1)
        (; pick = picks[j], eye, start, entry, refslot)
    end
    order = invperm(reduce(vcat, entries))
    return (; stacks = Tuple(stacks), order, refentry)
end

# Data and weights per correlation product at the measured values `keep`.
function _selfcal_weights(t::VisTable, keep)
    V, w = _dense(measurement(t)), _dense(weight(t))
    Vs, ws = _feedproducts(V), _feedproducts(w)
    wdata = map((v, wk) -> wk[keep] .* v[keep], Vs, ws)
    return wdata, map(wk -> wk[keep], ws)
end

# The correlation products of dense values: itself, or the four feed slabs `(…, 2, 2)`.
_feedproducts(x::AbstractArray{<:Number, 2}) = (x,)
_feedproducts(x::AbstractArray{<:Number, 4}) = ntuple(k -> x[:, :, mod1(k, 2), cld(k, 2)], Val(4))

_products(v::AbstractArray{<:Number}) = (v,)
_products(v::FieldDimArray{<:StaticArraysCore.StaticMatrix}) = ntuple(k -> fieldview(v, k), Val(4))

# φ̂ at every entry of the phase term, given the model visibilities without the station phase.
function _selfcal_solution(vt, J::ObservedSelfCalInstrumentModel)
    m = map(p -> p[J.keep], _products(vt))
    c = sum(map((w, mp) -> w .* conj.(mp), J.wdata, m))
    g = sum(map((w, mp) -> w .* abs2.(mp), J.weight, m))
    cx = vcat(zero(c[1:1]), c, conj.(c))
    gx = vcat(zero(g[1:1]), g, g)
    z = map(J.layout.stacks) do st
        A = cx[st.pick] .+ sum(gx[st.pick]; dims = 2) .* st.eye
        x = A[st.start]
        for _ in 2:J.iterations
            x = _matvec(A, _unitnorm(x))
        end
        return x[st.entry] .* conj.(x[st.refslot])
    end
    zo = reduce(vcat, z)[J.layout.order]
    return atan.(imag.(zo), real.(zo))
end

# `A[:, :, j] * x[:, j]` for every matrix of the stack `A`.
_matvec(A, x) = dropdims(sum(A .* reshape(x, 1, size(x)...); dims = 2); dims = 2)
_unitnorm(x) = x ./ sqrt.(sum(abs2, x; dims = 1))

@inline function apply_instrument(vis, J::ObservedSelfCalInstrumentModel, x)
    vt = apply_instrument(vis, J.model, x)
    φ = _selfcal_solution(vt, J) .+ parent(siteparams(getproperty(x, J.phase)))
    Δ = φ[J.entry1] .- φ[J.entry2]
    return _scaled(vt, complex.(cos.(Δ), sin.(Δ)))
end

_scaled(v::AbstractArray{<:Number}, r) = v .* r
_scaled(v::FieldDimArray{<:StaticArraysCore.StaticMatrix}, r) = FieldDimArray{SMatrix{2, 2}}(parent(v) .* r)

"""
    selfcal_phase(J, vis, θ)

The station phase `φ = φ̂ + residual` at the parameters `θ` of the instrument model `J`, bound
by [`observe`](@ref) and self-calibrating the station phase (see [`InstrumentModel`](@ref)),
for the model visibilities `vis`, as a `SiteArray`.
"""
function selfcal_phase(J, vis, θ)
    J isa ObservedSelfCalInstrumentModel ||
        throw(ArgumentError("the instrument model does not self-calibrate the station phase"))
    vt = apply_instrument(vis, J.model, θ)
    ε = siteparams(getproperty(θ, J.phase))
    return SiteArray(_selfcal_solution(vt, J) .+ parent(ε), times(ε), frequencies(ε), sites(ε))
end
