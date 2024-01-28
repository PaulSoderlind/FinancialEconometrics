# Introduction

This repository contains Julia code for a Financial Econometrics (MSc) course at UNISG.

This version (late January 2024) is a *major rewrite*. Although the topics and data sets are similar to before, the way the econometrics functions are called is changed. 


# Instructions

1.  Most files are jupyter notebooks. Click one of them to see it online. If GitHub fails to render the notebook or messes up the LaTeX in the Markdown cells, then use [nbviewer](https://nbviewer.jupyter.org/). Instructions: try to open the notebook at GitHub, copy the link and paste it in the address field of nbviewer.

2. To download this repository, use the Download (as zip) in the Github menu. Otherwise, clone it.

3. To get started, please check the *Ch00_HowToUse.ipynb* notebook first.


# On the Files

1. ChapterNumber_Topic.ipynb are notebooks organised around different topics. The chapter numbers correspond to the lecture notes (pdf), where more details are given (and the notation is explained).

2. Most statistical/econometric functions are organised in local modules, typically loaded at the top of the notebooks. The source code is in the jlFiles subfolder.

3. The pdf file contains the lecture notes.

4. The folder Data contains some data sets used in the notebooks.

5. The plots are in png format (so GitHub can show them). If you want sharper plots, change `default(fmt = :png)` to `default(fmt = :svg)` in one of the top cells.

5. The current version is tested on Julia 1.10.


# Relation to Other Julia Econometrics Codes

The notebooks are closely tied to my lecture notes. The focus is on learning, so most methods are built from scratch. For instance, to estimate a GARCH model, the notebook builds the likelihood function, calls on a routine for optimisation (for the point estimates) and then differentiation (for the standard errors).

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
