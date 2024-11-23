"""
    RidgeRegression(Y,X,λ,β₀=0)

Calculate ridge regression estimate with target vector `β₀`.
"""
function RidgeRegression(Y,X,λ,β₀=0)
    (T,K) = (size(X,1),size(X,2))
    isa(β₀,Number) && (β₀=fill(β₀,K))
    b = (X'X/T+λ*I)\(X'Y/T+λ*β₀)      #same as inv(X'X/T+λ*I)*(X'Y/T+λ*β₀)
    return b
end


"""
    StandardiseYX(Y0,X0)

Demean and make std=1 for `Y` and `X` (vector or matrices)

"""
function StandardiseYX(Y0,X0)
    Y = (Y0 .- mean(Y0,dims=1))./std(Y0,dims=1)
    X = (X0 .- mean(X0,dims=1))./std(X0,dims=1)
    return Y,X
end
