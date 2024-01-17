##------------------------------------------------------------------------------
"""
    OlsR2TestFn(R²,T,df)

Test of all slope coefficients. Notice that the regression must contain
an intercept for R² to be useful.

### Input
- `R²::Number`:    R² value
- `T::Int`:        number of observations
- `df::Number`:    number of (non-constant) regressors

### Output
- `RegrStat::Number`: test statistic
- `pval::Number`:     p-value

"""
function OlsR2TestFn(R²,T,df)
    RegrStat = T*R²/(1-R²)           #R\^2[TAB]
    pval     = 1 - cdf(Chisq(df),RegrStat)    #or ccdf() to get 1-cdf()
    return RegrStat, pval
end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    OlsWhitesTestFn(u,x)

Test of heteroskedasticity. Notice that the regression must contain 
an intercept for the test to be useful.

### Input
- `u::Vector`:   T-vector, residuals
- `x::Matrix`:   Txk, regressors

### Output
- `RegrStat::Number`: test statistic
- `pval::Number`:     p-value

"""
function OlsWhitesTestFn(u,x)

    (T,k) = (size(x,1),size(x,2))

    w = zeros(T,round(Int,k*(k+1)/2))   #matrix of cross products of x
    vv = 1
    for i = 1:k, j = i:k
        w[:,vv] = x[:,i].*x[:,j]        #eg. x1*x1, x1*x2, x2*x2
        vv        = vv + 1
    end

    R² = OlsGMFn(u.^2,w)[5]             #[5] picks out output 5
    df = rank(w) - 1                    #number of independent regressors in w

    WhiteStat = T*R²/(1-R²)
    pval      = 1 - cdf(Chisq(df),WhiteStat)

    return WhiteStat, pval

end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    OlsAutoCorrFn(u,L=1)

Test the autocorrelation of OLS residuals

### Input
- `u::Vector`:   T-vector, residuals
- `L::Int`:      scalar, number of lags in autocorrelation and Box-Pierce test

### Output
- `AutoCorr::Matrix`:   Lx3, autocorrelation, t-stat and p-value
- `BoxPierce::Matrix`:  1x2, Box-Pierce statistic and p-value
- `DW::Number`:         DW statistic

### Requires
- StatsBase, Distributions

"""
function OlsAutoCorrFn(u,L=1)

    T = size(u,1)

    Stdu = std(u)
    ρ    = autocor(u,1:L)        #\rho[TAB]
    t_ρ  = sqrt(T)*ρ             #t-stat of ρ 

    pval      = 2*(1.0 .- cdf.(Normal(0,1),abs.(t_ρ)))
    AutoCorr  = [ρ t_ρ pval]

    BPStat    = T*sum(ρ.^2)
    pval      = 1 - cdf(Chisq(L),BPStat)
    BoxPierce = [BPStat pval]

    DWStat    = mean(diff(u).^2)/Stdu^2

    return AutoCorr, BoxPierce, DWStat

end
##------------------------------------------------------------------------------
