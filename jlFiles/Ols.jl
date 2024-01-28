"""
    OlsGM(Y,X)

LS of Y on X; for one dependent variable, Gauss-Markov assumptions

### Input
- `Y::Vector`:    T-vector, the dependent variable
- `X::Matrix`:    Txk matrix of regressors (including deterministic ones)

### Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - yhat
- `Yhat::Vector`: T-vector, fitted values X*b
- `V::Matrix`:    kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsGM(Y,X)

    T    = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    σ²   = var(u)
    V    = inv(X'X)*σ²
    R²   = 1 - σ²/var(Y)

    return b, u, Yhat, V, R²

end


"""
    OlsNW(Y,X,m=0)

LS of Y on X; for one dependent variable, using Newey-West covariance matrix

### Input
- `Y::Vector`:    T-vector, the dependent variable
- `X::Matrix`:    Txk matrix of regressors (including deterministic ones)
- `m::Int`:       scalar, bandwidth in Newey-West

### Output
- `b::Vector`:    k-vector, regression coefficients
- `u::Vector`:    T-vector, residuals Y - Yhat
- `Yhat::Vector`: T-vector, fitted values X*b
- `V::Matrix`:    kxk matrix, covariance matrix of b
- `R²::Number`:   scalar, R² value

"""
function OlsNW(Y,X,m=0)

    T    = size(Y,1)

    b    = X\Y
    Yhat = X*b
    u    = Y - Yhat

    S    = CovNW(X.*u,m)         #Newey-West covariance matrix
    Sxx  = X'X
    V    = inv(Sxx)'S*inv(Sxx)     #covariance matrix of b
    R²   = 1 - var(u)/var(Y)

    return b, u, Yhat, V, R²

end


"""

    OLSyxReplaceNaN(Y,X)

Replaces any rows in Y and X with zeros if there is any NaN/missing in any of them.

"""
function OLSyxReplaceNaN(Y,X)

  vv = FindNNPs(Y,X)             #vv[t] = true if no missing/NaN i (y[t],x[t,:])

  (Yb,Xb)     = (copy(Y),copy(X))    #set both y[t] and x[t,:] to 0 if any missing/NaN for obs. t
  Yb[.!vv]   .= 0
  Xb[.!vv,:] .= 0

  return vv, Yb, Xb

end
