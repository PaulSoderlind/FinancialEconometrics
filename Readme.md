# Introduction

This repository contains Julia code for a Financial Econometrics (MSc) course at UNISG. 


# Instructions

1.  Most files are jupyter notebooks. Click one of them to see it online. If GitHub fails to render the notebook, then use [nbviewer](https://nbviewer.jupyter.org/). Instructions: try to open the notebook at GitHub, copy the link and paste it in the address field of nbviewer.

2.  To download this repository, use the Download (as zip) in the Github menu. Otherwise, clone it.


# On the Files

1. ChapterNumber_Topic.ipynb are notebooks organised around different topics. The chapter numbers correspond to the lecture notes (pdf), where more details are given (and the notation is explained).

2. The pdf file contains the lecture notes.

3. The folder Data contains some data sets used in the notebooks, while the folder jlFiles contains .jl files with some functions (also used in the notebooks).

4. The current version is tested on Julia 1.5 and 1.6.


# Relation to Other Julia Econometrics Codes

The notebooks are closely tied to my lecture notes. The focus is on learning, so most methods are built up from scratch. For instance, to estimate a GARCH model, the notebook builds the likelihood function, calls on a routine for optimisation (for the point estimates) and then differentiation (for the standard errors).

See [Michael Creel's code](https://github.com/mcreel/Econometrics)
for a similar approach (also focused on teaching)

The following packages provide more convenient (and often more powerful) routines:  

[GLM.jl](https://github.com/JuliaStats/GLM.jl)
for regressions

[CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl)
for robust (heteroskedasticity and/or autocorrelation) covariance estimates

[HypothesisTests.jl](https://github.com/JuliaStats/HypothesisTests.jl)
for testing residuals and distributions

[ARCHModels.jl](https://github.com/s-broda/ARCHModels.jl)
for estimating ARCH and GARCH models

[KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl)
for kernel density estimation

[QuantileRegressions.jl](https://github.com/pkofod/QuantileRegressions.jl)
for quantile regressions
