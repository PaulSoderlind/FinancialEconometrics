module FinEcmt_OLS

using Statistics, LinearAlgebra, DelimitedFiles, Distributions, StatsBase
import Printf
using FiniteDiff: finite_difference_jacobian as jacobian

export BinScatter, BinDummies,
BeraJarqueTest, CompanionFormAR, CovNW, CovToCor,
DeltaMethod, DrawBlocks, EMA, EWportf,
FindNNPanel, FindNN, FirstDiff,
IndividualDemean, KolSmirTest,
NWCovPs, OLSyxReplaceNaN, OlsAutoCorr, OlsBasic, OlsGM, OlsNW, RegressionTable,
OlsR2Test, OlsSure, OlsWhitesTest,RegressionFit, VIF, DiagnosticsTable,
PanelOls, PanelReshuffle, DummiesCreate, TimeDummiesTTN,
FixedIndivEffects,FixedTimeEffects, FixedIndivTimeEffects, FixedTimeIndivEffects, FWonZRepeated!,
PutDataInNT, Readcsv,
QuantRegrIRLS, ReturnStats, RidgeRegression, StandardiseYX,
TwoSLS, excise, lag,
printblue, printlnPs, printmagenta, printmat, printred, printyellow, @doc2,
rankPs, sortLoHi

include("BinScatter.jl")
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