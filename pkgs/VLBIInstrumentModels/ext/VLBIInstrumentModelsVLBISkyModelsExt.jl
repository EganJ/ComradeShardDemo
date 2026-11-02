module VLBIInstrumentModelsVLBISkyModelsExt

using VLBIInstrumentModels
import VLBIInstrumentModels as VIM
using VLBIInstrumentModels: VisTable, _measuredvalues, _measuredpoints, measurement, domain
using VLBISkyModels
using ComradeBase: ComradeBase, StructuredDomain, executor
import DimensionalData as DD

"""
    FourierDualDomain(grid::AbstractRectiGrid, t::VisTable, alg::FourierTransform; executor = executor(grid))

The dual domain of the image `grid` and the points of the table `t`: its baselines in
wavelengths and the times of its records, transformed with `alg`. The model visibilities of a
sky model on it are those `apply_instrument` and `simulate_observation` take for `t`.
"""
function VLBISkyModels.FourierDualDomain(grid::ComradeBase.AbstractRectiGrid, t::VisTable, alg::VLBISkyModels.FourierTransform; executor = executor(grid))
    return FourierDualDomain(grid, _skydomain(domain(t), executor), alg)
end

# The baselines in wavelengths, precomputed, and the times of the table's records, without the
# station coordinates only the instrument reads.
function _skydomain(d::StructuredDomain, executor)
    return StructuredDomain(
        DD.dims(d); executor, header = ComradeBase.header(d),
        U = collect(d.U) => (:Node, :Fr), V = collect(d.V) => (:Node, :Fr), Ti = d.Ti => (:Node,),
    )
end

function VIM.dirty_image(fov::Real, npix::Int, t::VisTable)
    keep = _dirtyvalues(t)
    return _dirty(fov, npix, t, keep, baseimage(measurement(t))[keep])
end

function VIM.dirty_beam(fov::Real, npix::Int, t::VisTable)
    keep = _dirtyvalues(t)
    return _dirty(fov, npix, t, keep, fill(complex(oneunit(fov), zero(fov)), length(keep)))
end

function _dirtyvalues(t::VisTable)
    eltype(measurement(t)) <: Complex ||
        throw(ArgumentError("a dirty image is formed from a table of complex visibilities, got one of $(eltype(measurement(t)))"))
    return _measuredvalues(t)
end

# Each baseline and its reflection, with the conjugate visibility, so the transform is real.
function _dirty(fov, npix, t, keep, vis)
    d = domain(t)
    U, V = d.U[keep], d.V[keep]
    (; Ti, Fr) = _measuredpoints(t)
    points = StructuredDomain((U = vcat(U, -U), V = vcat(V, -V), Ti = vcat(Ti, Ti), Fr = vcat(Fr, Fr)))
    g = spatialgrid(fov, fov, npix, npix)
    plr = VLBISkyModels.reverse_plan(FourierDualDomain(g, points, NFFTAlg()))
    m = plr.plan * (conj.(vcat(vis, conj.(vis))) .* plr.phases)./npix^2
    return IntensityMap(real.(m), g)
end

end
