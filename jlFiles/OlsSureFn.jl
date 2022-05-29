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
- `V::Matrix`:     covariance matrix of vec(b)
- `R²::Vector`:    n-vector, R² values

"""
function OlsSureFn(Y,X,NWQ=false,m=0)

    (T,n) = (size(Y,1),size(Y,2))
    k     = size(X,2)

    b     = X\Y
    Yhat  = X*b
    u     = Y - Yhat

    g = repeat(X,outer=(1,n)).*repeat(u,inner=(1,k))  #[u[:,1].*X,u[:,2].*X...]

    Sxx = X'X
    if NWQ
        S     = CovNWFn(g,m)            #Newey-West covariance matrix
        Sxx_1 = kron(I(n),inv(Sxx))
        V     = Sxx_1 * S * Sxx_1
    else
        V = kron(cov(u),inv(Sxx))      #traditional covariance matrix, Gauss-Markov
    end

    R²   = 1 .- var(u,dims=1)./var(Y,dims=1)

    return b, u, Yhat, V, R²

end
#------------------------------------------------------------------------------
