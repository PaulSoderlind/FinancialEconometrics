"""
    DeltaMethod(fn::Function,β,V,x=NaN)

Apply the delta method

# Input
- `β::Vector`:        with parameters 
- `V::Matrix`:        Cov(β)
- `x::VecOrMat`:      data (if any is needed)

# Requires
- `using FiniteDiff: finite_difference_jacobian as jacobian`

"""
function DeltaMethod(fn::Function,β,V,x=NaN)
    P = jacobian(b->fn(b,x),β)        #Numerical Jacobian
    Cov_fn = P*V*P'
    return Cov_fn
end
