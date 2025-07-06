module FinEcmt_TimeSeries

using Statistics, LinearAlgebra

export garch11LL, egarch11LL, DccLL, DccParTrans,
EWMA_variance, EWMA_covariance, Dcc_EWMA,
EMA, ARMAFilter, VARFilter,
CompanionFormAR, ARpEst, CompanionFormVAR, VAR1IRF, VAR1AutoCov, VAR1Forecast,
MAqLL, MAqToAutocorr, YuleWalker


include("Garch.jl")
include("Garch_EWMA.jl")
include("TimeSeriesFilter.jl")
include("TimeSeries_VAR.jl")
include("TimeSeries_MA.jl")
include("UtilityFunctions.jl")

end
