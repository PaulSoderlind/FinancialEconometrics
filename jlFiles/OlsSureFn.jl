"""
    OlsSureFn(Y,X,NWQ=false,m=0)

LS of Y on X; where Y is Txn, and X is the same for all regressions

# Usage
(b,res,Yhat,Covb,R²) = OlsSureFn(Y,X,NWQ,m)

# Input
- `Y::Matrix`:     Txn, the n dependent variables
- `X::Matrix`:     Txk matrix of regressors (including deterministic ones)
- `NWQ:Bool`:      if true, then Newey-West's covariance matrix is used, otherwise Gauss-Markov
- `m::Int`:        scalar, bandwidth in Newey-West

# Output
- `b::Matrix`:     n*kx1, regression coefficients
- `u::Matrix`:     Txn, residuals Y - Yhat
- `Yhat::Matrix`:  Txn, fitted values X*b
- `V::Matrix`:     covariance matrix of θ=vec(b)
- `R²::Matrix`:    1xn matrix, R² values

"""
function OlsSureFn(Y,X,NWQ=false,m=0)

    (T,n) = (size(Y,1),size(Y,2))
    k     = size(X,2)

    b     = X\Y
    Yhat  = X*b
    u     = Y - Yhat

    Sxx = X'X

    if NWQ
        g      = hcat([X.*u[:,i] for i=1:n]...)    #hcat(X.*u[:,1],X.*u[:,2], etc)
        S      = CovNWFn(g,m)           #Newey-West covariance matrix
        SxxM_1 = kron(I(n),inv(Sxx))    #or
        V      = SxxM_1 * S * SxxM_1
    else
        V = kron(cov(u),inv(Sxx))      #traditional covariance matrix, Gauss-Markov
    end

    R²   = 1 .- var(u,dims=1)./var(Y,dims=1)

    return b, u, Yhat, V, R²

end
