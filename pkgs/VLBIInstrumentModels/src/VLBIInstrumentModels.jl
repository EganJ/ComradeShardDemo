module VLBIInstrumentModels

using Accessors: getproperties, setproperties
using Adapt
using ChainRulesCore
import Distributions as Dists
using DocStringExtensions
using EnzymeCore
using EnzymeCore: EnzymeRules
using IntervalSets
using LinearAlgebra
import ProbabilityTransports as PT
using Random
using Reexport
using SparseArrays
using StaticArraysCore
using StaticArraysCore: SMatrix, SVector
using StructArrays: StructVector, StructArray
import StructArrays
using Tables
import TransformVariables as TV
import TransformVariables: transform, inverse, dimension
import VLBIImagePriors: asflat, ascube
using VLBIImagePriors
using VLBILikelihoods: ComplexVisLikelihood, CoherencyLikelihood
using ReactantCore
import DimensionalData as DD
using Dates: Dates, Date, DateTime, datetime2julian
import Measurements
import Printf

@reexport using ComradeBase
@reexport using PolarizedTypes
using ComradeBase: AbstractHeader, rgetindex, rsetindex!

export Ant

# The station dim of per-station quantities, e.g. instrument terms or geometry.
DD.@dim Ant "station"

include("macro_utils.jl")
include("mounts.jl")
include("observations/observations.jl")
include("instrument/instrument.jl")
include("visualizations/data.jl")
include("visualizations/fields.jl")
include("visualizations/makie.jl")

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :AbstractInstrumentModel, :AbstractJonesMatrix,
            :CalTable, :SiteLookup, :Segmentation,
            :timestamps, :gauge_pins,
        )
    )
end

end
