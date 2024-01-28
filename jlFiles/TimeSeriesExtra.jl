"""
    CompanionFormAR(a)

Companion form of AR(p), that is, write an AR(p) as a VAR(1)
"""
function CompanionFormAR(a)
    p = length(a)
    if p > 1                      #if order > 1
        A1 = vcat(a',[I  zeros(p-1)])
    else                          #if already VAR(1)
        A1 = copy(a)
    end
    return A1
end


"""
    ARpEst(y,p)

Estimate an AR(p) model (with an intercept) on the data in a vector `y`.

Output: the slope coefficients (not the intercept).

"""
function ARpEst(y,p)

    T    = length(y)

    xAll = fill(NaN,T,p)               #build matrix of regressors
    for s = 1:p                        #or reduce(hcat,lag(y,s) for s=1:p)
        xAll[:,s] = lag(y,s)
    end
    xAll = [xAll ones(T)]              #add constant last
    #printmat([y xAll][1:10,:])        #uncomment to see the regressors

    b = xAll[p+1:end,:]\y[p+1:end]     #OLS, cut the first p observatioms
    a = b[1:end-1]                     #slopes

    return a

end


"""
    MAqLL(par,y)

Log likelihood function for MA(q) process.

"""
function MAqLL(par::Vector,y)
    (θ,σ) = (par[1:end-1],par[end])
    q     = length(θ)
    ϵ     = ARMAFilter(y,-θ)           #ϵ is AR(q) with coefs -θ
    LL_i  = -1/2*log(2*π) .- 1/2*log(σ^2) .- 1/2*ϵ.^2/σ^2
    LL    = sum(LL_i)                  #sum(log-likelihood values)
    return LL, ϵ
end
