#------------------------------------------------------------------------------
"""
    OlsGMFn(Y,X)

LS of Y on X; for one dependent variable, Gauss-Markov assumptions

# Usage
(b,u,Yhat,V,R²) = OlsGMFn(Y,X)

# Input
- `Y::Vector`:    T-vector, the dependent variable
- `X::Matrix`:    Txk matrix of regressors (including deterministic ones)

# Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - yhat
- `Yhat::Vector`: T-vector, fitted values X*b
- `V::Matrix`:    kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsGMFn(Y,X)

    T    = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    σ²   = var(u)
    V    = inv(X'X)*σ²
    R²   = 1 - σ²/var(Y)

    return b, u, Yhat, V, R²

end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    OlsNWFn(Y,X,m=0)

LS of Y on X; for one dependent variable, using Newey-West covariance matrix

# Usage
(b,u,Yhat,V,R²) = OlsNWFn(Y,X,m)

# Input
- `Y::Array`:     Tx1, the dependent variable
- `X::Array`:     Txk matrix of regressors (including deterministic ones)
- `m::Int`:       scalar, bandwidth in Newey-West

# Output
- `b::Array`:     kx1, regression coefficients
- `u::Array`:     Tx1, residuals Y - Yhat
- `Yhat::Vector`: Tx1, fitted values X*b
- `V::Array`:     kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsNWFn(Y,X,m=0)

    T    = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    S    = CovNWFn(X.*u,m)         #Newey-West covariance matrix
    Sxx  = X'X
    V    = inv(Sxx)'S*inv(Sxx)     #covariance matrix of b
    R²   = 1 - var(u)/var(Y)

    return b, u, Yhat, V, R²

end
#------------------------------------------------------------------------------
