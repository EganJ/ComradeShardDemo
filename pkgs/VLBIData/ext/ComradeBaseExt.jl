module ComradeBaseExt

using VLBIData
using VLBIData.StructArrays
using VLBIData.Unitful
using VLBIData.DataManipulation
using ComradeBase


# Point maps are `UnstructuredMap`s up to ComradeBase 0.9 and `StructuredMap`s from 0.10.
const PointMap = isdefined(ComradeBase, :UnstructuredMap) ? ComradeBase.UnstructuredMap : ComradeBase.StructuredMap

function VLBI.uvtable(usm::PointMap)
	ndims(usm) == 1 || error("uvtable needs a map over a single point dim, got a map of size $(size(usm))")
	hasproperty(usm, :U) && hasproperty(usm, :V) || error("UnstructuredMap must have U and V properties, got $(propertynames(usm))")
	nt = (;spec = UV.(usm.U, usm.V), value = Vector(usm))
	if hasproperty(usm, :Fr)
		nt = (; nt..., freq_spec = usm.Fr .* u"Hz" .|> u"GHz")
	end
	if hasproperty(usm, :Ti)
		nt = (; nt..., datetime = usm.Ti)
	end
	return StructArray(nt)
end

end