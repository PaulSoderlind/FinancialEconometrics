"""
    CovNWFn(g0,m=0,DivideByT=0)

Calculates covariance matrix of sample sum (DivideByT=0), scaled average (DivideByT=1) or
average (DivideByT=2).


### Input
- `g0::Matrix`:      Txq Mat#rix of q moment conditions
- `m::Int`:          number of lags to use
- `DivideByT::Int`:  divide the result by T^DivideByT

### Output
- `S::Matrix`: qxq covariance matrix

### Remark
- `DivideByT=0`: Var(x_1+x_2+...X_T)
- `DivideByT=1`: Var((x_1+x_2+...X_T)/sqrt(T))
- `DivideByT=2`: Var((x_1+x_2+...X_T)/T)


"""
function CovNWFn(g0,m=0,DivideByT=0)

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
