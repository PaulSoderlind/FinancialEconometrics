module FinEcmt_TimeSeries

using Statistics, LinearAlgebra

export ARMAFilter,  VARFilter,
CompanionFormAR, ARpEst, MAqLL,
garch11LL, egarch11LL, DccLL, DccParTrans

include("Garch.jl")
include("TimeSeriesExtra.jl")
include("TimeSeriesFilter.jl")
include("UtilityFunctions.jl")

end
