module FinEcmt_KernelRegression

using Statistics

export EpanechnikovKernel, GaussianKernel, KernelDensity, KernelRegression,
LocalLinearRegression, UniformKernel, hRuleOfThumb

include("CovNW.jl")
include("KernelRegression.jl")
include("Ols.jl")

end
