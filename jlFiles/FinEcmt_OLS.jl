module FinEcmt_OLS

using Statistics, LinearAlgebra, Distributions, StatsBase
import Printf
using FiniteDiff: finite_difference_jacobian as jacobian

export BeraJarqueTest, CompanionFormAR, CovNW, CovToCor,
DeltaMethod, DrawBlocks, EMA, EWportf,
FindNNPanel, FindNNPs, FirstDiff, FixedEffects,
IndividualDemean, KolSmirTest,
NWCovPs, OLSyxReplaceNaN, OlsAutoCorr, OlsGM, OlsNW, OlsR2Test, OlsSure, OlsWhitesTest,
PanelOls, PanelyxReplaceNaN, PanelyxReplaceNaN!, PanelyxReshuffle, PutDataInNT,
QuantRegrIRLS, ReturnStats, RidgeRegression, StandardiseYX,
TwoSLS, excise, lag,
printblue, printlnPs, printmagenta, printmat, printred, printyellow, @doc2,
rankPs, sortLoHi

include("CovNW.jl")
include("DeltaMethod.jl")
include("DistributionTests.jl")
include("Ols.jl")
include("OlsDiagnostics.jl")
include("OlsSure.jl")
include("PanelOls.jl")
include("PanelRegrBasic.jl")
include("PortfolioSorts.jl")
include("printmat.jl")
include("QuantRegrIRLS.jl")
include("RidgeRegression.jl")
include("SimulationFunctions.jl")
include("TwoSLS.jl")
include("UtilityFunctions.jl")

end
