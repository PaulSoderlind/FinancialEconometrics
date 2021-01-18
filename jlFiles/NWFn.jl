"""
    CovNWFn(g0,m=0)

Calculates covariance matrix of sample average.

# Input
- `g0::Matrix`: Txq Matrix of q moment conditions
- `m:int`:     scalar, number of lags to use

# Output
- `S::Matrix`: qxq covariance matrix(average g0)

"""
function CovNWFn(g0,m=0)

    T = size(g0,1)                    #g0 is Txq
    m = min(m,T-1)                    #number of lags

    g = g0 .- mean(g0,dims=1)         #normalizing to zero means

    S = g'g                           #(qxT)*(Txq)
    for s = 1:m
        Λ_s = g[s+1:T,:]'g[1:T-s,:]   #same as Sum[g_t*g_{t-s}',t=s+1,T]
        S   = S  +  (1 - s/(m+1))*(Λ_s + Λ_s')
    end

    return S

end
