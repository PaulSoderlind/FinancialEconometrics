"""
    CovNW(g0,m=0,DivideByT=0)

Calculates covariance matrix of sample sum (DivideByT=0), √T*(sample average) (DivideByT=1)
or sample average (DivideByT=2).


### Input
- `g0::Matrix`:      Txq matrix of data
- `m::Int`:          number of lags to use
- `DivideByT::Int`:  divide the result by T^DivideByT

### Output
- `S::Matrix`: qxq covariance matrix

### Remark
- `DivideByT=0`: Var(g₁+g₂+...), variance of sample sum
- `DivideByT=1`: Var(g₁+g₂+...)/T = Var(√T gbar), where gbar is the sample average. This is
   the same as Var(gᵢ) if data is iid
- `DivideByT=2`: Var(g₁+g₂+...)/T^2 = Var(gbar)


"""
function CovNW(g0,m=0,DivideByT=0)

    T = size(g0,1)                    #g0 is Txq
    m = min(m,T-1)                    #number of lags

    g = g0 .- mean(g0,dims=1)         #normalizing to zero means

    S = g'g                           #(qxT)*(Txq)
    for s = 1:m
        Λ_s = g[s+1:T,:]'g[1:T-s,:]   #same as Sum[g_t*g_{t-s}',t=s+1,T]
        S   = S  +  (1 - s/(m+1))*(Λ_s + Λ_s')
    end

    (DivideByT > 0) && (S = S/T^DivideByT)

    return S

end
