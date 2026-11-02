module FieldDimArraysAbstractFFTsExt

using AbstractFFTs: AbstractFFTs, Plan, ScaledPlan, AdjointPlan
using FieldDimArrays: FieldDimArray, _rewrap

# A plan for a `FieldDimArray` acts on its parent and transforms every component.
for f in (:plan_fft, :plan_bfft, :plan_ifft, :plan_fft!, :plan_bfft!, :plan_ifft!)
    @eval function AbstractFFTs.$f(a::FieldDimArray, region; kws...)
        _checkregion(a, region)
        return AbstractFFTs.$f(parent(a), region; kws...)
    end
end

function _checkregion(a::FieldDimArray, region)
    issubset(region, 1:ndims(a)) ||
        throw(ArgumentError("region $region is not within the $(ndims(a)) dims of the FieldDimArray"))
    return nothing
end

for P in (Plan, ScaledPlan, AdjointPlan)
    @eval Base.:*(p::$P, a::FieldDimArray) = _rewrap(a, p * parent(a))
end

end
