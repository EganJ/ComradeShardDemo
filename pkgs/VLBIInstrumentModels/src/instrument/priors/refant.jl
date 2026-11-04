abstract type ReferencingScheme end

export NoReference, SingleReference, MultiReference, SensitiveReference

struct NoReference <: ReferencingScheme end

"""
    SingleReference(site::Symbol, val)

Selects a single reference site for all scans. The value of the site is set to `val`.
"""
struct SingleReference{T} <: ReferencingScheme
    site::Symbol
    value::T
end


"""
    MultiReference(sites, val)

Selects several reference sites, fixing every entry of each site in `sites` to `val`.
`sites` is any iterable of site names.

Use it for a parameter with more than one gauge freedom. The usual case is a
track-segmented phase offset when the schedule splits the array into groups that share no
station, for example when a subset of the array observes for a stretch without the
reference site. Each such group needs its own reference.

## Notes
This pins the sites it is given and does not check that the result leaves the model
identified. Too few pins leave a flat direction in the posterior; too many
over-constrain a scan, since a scan carries only one global phase.
"""
struct MultiReference{S <: AbstractVector{Symbol}, T} <: ReferencingScheme
    sites::S
    value::T
    function MultiReference{S, T}(sites, value) where {S, T}
        ss = convert(S, sites)::S
        isempty(ss) && throw(ArgumentError("MultiReference needs at least one site"))
        allunique(ss) ||
            throw(ArgumentError("MultiReference sites must be distinct, got $ss"))
        return new{S, T}(ss, value)
    end
end

MultiReference{S}(sites, value) where {S} = MultiReference{S, typeof(value)}(sites, value)
MultiReference(sites::AbstractVector{Symbol}, value) =
    MultiReference{typeof(sites), typeof(value)}(sites, value)
MultiReference(sites, value) = MultiReference(Symbol[Symbol(s) for s in sites], value)

#    EntryReference(indices, value)
#
# Pins the parameter entries at `indices`, positions in the parameter's `SiteLookup`, to
# `value`. The indices are meaningful only for the `SiteLookup` they were computed from.
# Internal type.
struct EntryReference{I <: AbstractVector{Int}, T} <: ReferencingScheme
    indices::I
    value::T
    function EntryReference{I, T}(indices, value) where {I, T}
        ii = convert(I, indices)::I
        allunique(ii) ||
            throw(ArgumentError("EntryReference indices must be distinct, got $ii"))
        return new{I, T}(ii, value)
    end
end

EntryReference{I}(indices, value) where {I} = EntryReference{I, typeof(value)}(indices, value)
EntryReference(indices::AbstractVector{Int}, value) =
    EntryReference{typeof(indices), typeof(value)}(indices, value)
EntryReference(indices, value) = EntryReference(collect(Int, indices), value)

#    CompositeReference(schemes...)
#
# Applies every scheme in `schemes` to the same parameter and pins the union of their
# indices. An index two schemes both claim must be given the same value by both. Internal
# type.
struct CompositeReference{S <: Tuple} <: ReferencingScheme
    schemes::S
end

CompositeReference(schemes::ReferencingScheme...) = CompositeReference(schemes)

# The value a scheme pins its entries to; `nothing` when it pins nothing.
reference_value(::NoReference) = nothing
reference_value(r::ReferencingScheme) = r.value
function reference_value(r::CompositeReference)
    for s in r.schemes
        v = reference_value(s)
        v === nothing || return v
    end
    return nothing
end

"""
    SensitiveReference(val)

Selects, in each time and frequency segment of the parameter, the site with the largest total
data weight `Σ 1/σ²` over its measured values in that segment, and fixes its entry to `val`.
"""
struct SensitiveReference{T} <: ReferencingScheme
    value::T
end

# Only SensitiveReference reads the visibility table.
reference_indices(_, st::SiteLookup, ::NoReference) = [], nothing
function reference_indices(_, st::SiteLookup, p::SingleReference)
    inds = findall(==(p.site), st.sites)
    return inds, fill(p.value, length(inds))
end

function reference_indices(_, st::SiteLookup, p::MultiReference)
    inds = Int[]
    for s in p.sites
        si = findall(==(s), st.sites)
        # An absent site would pin nothing and leave the gauge unfixed.
        isempty(si) && throw(
            ArgumentError(
                "MultiReference site $s is not among this parameter's sites " *
                    "$(sort(unique(st.sites)))"
            )
        )
        append!(inds, si)
    end
    sort!(inds)
    return inds, fill(p.value, length(inds))
end

function reference_indices(_, st::SiteLookup, r::EntryReference)
    inds = sort(r.indices)
    for i in inds
        checkindex(Bool, eachindex(st.sites), i) || throw(
            ArgumentError(
                "EntryReference index $i is outside this parameter's " *
                    "$(length(st.sites)) entries"
            )
        )
    end
    return inds, fill(r.value, length(inds))
end

function reference_indices(t, st::SiteLookup, r::CompositeReference)
    parts = [reference_indices(t, st, s) for s in r.schemes]
    filter!(p -> !isempty(first(p)), parts)
    isempty(parts) && return Int[], nothing
    T = mapreduce(p -> eltype(last(p)), promote_type, parts)
    inds = Int[]
    vals = T[]
    for (is, vs) in parts, (i, v) in zip(is, vs)
        j = findfirst(==(i), inds)
        if j === nothing
            push!(inds, i)
            push!(vals, v)
        elseif vals[j] != v
            throw(
                ArgumentError(
                    "CompositeReference pins entry $i (site $(st.sites[i])) to both " *
                        "$(vals[j]) and $v"
                )
            )
        end
    end
    p = sortperm(inds)
    return inds[p], vals[p]
end

function reference_indices(tab::VisTable, st::SiteLookup, r::SensitiveReference)
    w = _entryweights(tab, st)
    groups = Dict{Tuple{eltype(st.times), eltype(st.frequencies)}, Vector{Int}}()
    for i in eachindex(st.sites, st.times, st.frequencies)
        push!(get!(groups, (st.times[i], st.frequencies[i]), Int[]), i)
    end
    fixedinds = sort!([inds[argmax(w[inds])] for inds in values(groups)])
    return fixedinds, fill(r.value, length(fixedinds))
end

# The total data weight of the measured values each entry of `st` covers: those of its site
# whose time and frequency lie in the entry's segments.
function _entryweights(t::VisTable, st::SiteLookup)
    p = _measuredpoints(t)
    wv = _valueweights(t)[_measuredvalues(t)]
    bysite = Dict{Symbol, Vector{Int}}()
    for i in eachindex(st.sites)
        push!(get!(bysite, st.sites[i], Int[]), i)
    end
    w = zeros(Float64, axes(st.sites))
    for k in eachindex(wv, p.Ti, p.Fr, p.sites), s in p.sites[k], i in get(bysite, s, Int[])
        (p.Ti[k] in st.times[i] && p.Fr[k] in st.frequencies[i]) && (w[i] += wv[k])
    end
    return w
end
