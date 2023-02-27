"""
    MLE(LLtFun,par0,x,lower,upper)

Calculate ML point estimates of K parameters and three different types of standard errors:
from the Information matrix, from the gradients and the sandwich approach.

# Input
- `LLtFun::Function`:  name of log-likelihood function
- `par0::Vector`:      K-vector, starting guess of the parameters
- `x::VecOrMat`:       vector or matrix with data, unpacked (if necessary) inside LLtFun
- `lower::Vector`:     lower bounds on the parameters, nothing or fill(-Inf,K) if no bounds
- `upper::Vector`:     upper bounds on the parameters, nothing or fill(Inf,K) if no bounds

# Requires
- using FiniteDiff: finite_difference_hessian as hessian, finite_difference_jacobian as jacobian

"""
function MLE(LLtFun,par0,x,lower=nothing,upper=nothing)


    LL_t = LLtFun(par0,x)
    T    = length(LL_t)

    NoBounds = (isnothing(lower) && isnothing(upper)) || (all(!isfinite,lower) && all(!isfinite,upper))

    if NoBounds
        Sol = optimize(par->-sum(LLtFun(par,x)),par0) #minimize -sum(LLt)
    else
        Sol = optimize(par->-sum(LLtFun(par,x)),lower,upper,par0)
    end
    parHat = Optim.minimizer(Sol)                    #the optimal solution

    Ia = -hessian(par->mean(LLtFun(par,x)),parHat)   #2nd derivatives of mean(LLt)
    Ia       = (Ia+Ia')/2                            #to guarantee symmetry
    vcv      = inv(Ia)/T
    std_hess = sqrt.(diag(vcv))

    δL       = jacobian(par->LLtFun(par,x),parHat)   #TxK
    J        = δL'δL/T                               #KxT * TxK
    vcv      = inv(J)/T
    std_grad = sqrt.(diag(vcv))

    vcv       = inv(Ia) * J * inv(Ia)/T
    std_sandw = sqrt.(diag(vcv))                     #std from sandwich

   return parHat, std_hess, std_grad, std_sandw

end
