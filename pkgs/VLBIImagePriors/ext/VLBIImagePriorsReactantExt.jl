module VLBIImagePriorsReactantExt

using VLBIImagePriors
using Reactant
using ComradeBase
using LinearAlgebra
using FFTW

using Reactant: AnyTracedRArray, RNumber

# One convolution, images along the batch dimension; κ² stays out of the kernel so it is constant.
function VLBIImagePriors.gmrf_qv(I::AnyTracedRArray{<:Any, 3}, κ²)
    T = Reactant.unwrapped_eltype(I)
    nx, ny, nb = size(I)
    x = Reactant.Ops.reshape(Reactant.materialize_traced_array(I), [nx, ny, 1, nb])
    stencil = Reactant.Ops.constant(reshape(T[0 -1 0; -1 4 -1; 0 -1 0], 3, 3, 1, 1))
    lx = Reactant.Ops.convolution(
        [nx, ny, 1, nb], x, stencil;
        input_batch_dim = 4, input_feature_dim = 3, input_spatial_dims = [1, 2],
        kernel_input_dim = 3, kernel_output_dim = 4, kernel_spatial_dims = [1, 2],
        output_batch_dim = 4, output_feature_dim = 3, output_spatial_dims = [1, 2],
        padding = [1 1; 1 1], feature_group_count = 1, batch_group_count = 1,
    )
    return Reactant.Ops.reshape(lx, [nx, ny, nb]) .+ κ² .* I
end

# This is currently required because Reactant has a performance issue with mapreduce
# on non-traced arrays
function LinearAlgebra.logdet(d::MarkovRandomFieldGraph{N}, ρ::RNumber) where {N}
    κ² = VLBIImagePriors.κ(ρ, Val(N))^2
    tmp = log.(κ² .+ d.λQ)
    a = sum(tmp)
    return N * a - length(d.λQ) * log(VLBIImagePriors.mrfnorm(d, κ²))
end

as(ps::VLBIImagePriors.AbstractPowerSpectrum, kx, ky) = VLBIImagePriors.ampspectrum(ps, (kx, ky))
function VLBIImagePriors._spectrum!(::ComradeBase.ReactantEx, ns::Reactant.AnyTracedRArray, ps::VLBIImagePriors.AbstractPowerSpectrum, kx, ky)
    return ns .= as.(Ref(ps), kx, ky')
end

end
