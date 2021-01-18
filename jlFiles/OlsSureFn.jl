"""
    OlsSureFn(Y,X,m=0)

LS of Y on X; where Y is Txn, and X is the same for all regressions

# Usage
(b,res,Yhat,Covb,R2) = OlsSureFn(Y,X,m)

# Input
- `Y::Matrix`:     Txn, the n dependent variables
- `X::Matrix`:     Txk matrix of regressors (including deterministic ones)
- `m::Int`:        scalar, bandwidth in Newey-West

# Output
- `b::Matrix`:     n*kx1, regression coefficients
- `u::Matrix`:     Txn, residuals Y - Yhat
- `Yhat::Matrix`:  Txn, fitted values X*b
- `V::Matrix`:     covariance matrix of vec(b)
- `R2::Vector`:    n-vector, R2 values

"""
function OlsSureFn(Y,X,m=0)

    (T,n) = (size(Y,1),size(Y,2))
    k     = size(X,2)

    b     = X\Y
    Yhat  = X*b
    u     = Y - Yhat

    g     = zeros(T,n*k)
    for i = 1:n
      vv      = (1+(i-1)*k):(i*k)   #1:k,(1+k):2k,...
      g[:,vv] = X.*u[:,i]           #moment conditions for Y[:,i] regression
    end

    S0    = CovNWFn(g,m)            #Newey-West covariance matrix
    Sxxi  = -X'X
    Sxx_1 = kron(I(n),inv(Sxxi))
    V     = Sxx_1 * S0 * Sxx_1

    R2   = 1.0 .- var(u,dims=1)./var(Y,dims=1)

    return b, u, Yhat, V, R2

end
