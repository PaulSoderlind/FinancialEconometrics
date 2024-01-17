##------------------------------------------------------------------------------
"""
    DrawBlocksFn(T,BlockSize)

Draw a T-vector of indices `v` that can be used to create bootstrap residuals. 
The indices are such that they form blocks of length `BlockSize`.

"""
function DrawBlocksFn(T,BlockSize)
    nBlocks = cld(T,BlockSize)                 #number of blocks, rounded up
    v0      = rand(1:T,nBlocks)                #nBlocks, random starting obs of blocks
    v       = vec(v0' .+ vec(0:BlockSize-1))   #each block in a column
    v       = replace(z -> z>T ? z-T : z,v)    #wrap around if index > T
    #println(v)                                #uncomment to see result
    return v
end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    GarchSim(T,ω,α,β)

Simulate a time series of T residuals from a GARCH(1,1) process.

### Remark
- The vector of σ² values is not exported from the function. If needed, this could easily be changed.

"""
function GarchSim(T,ω,α,β)

    (σ²,u) = [zeros(T) for i=1:2]
    σ²[1]  = ω/(1-α-β)                              #average σ² as starting value
    for t = 2:T
        σ²[t] = ω + α*u[t-1]^2 + β*σ²[t-1]
        u[t]  = sqrt(σ²[t])*randn()
    end

    return u

end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    SimOLS(NSim,T,α)

Simulate data for a regression model with heteroskedastic errors and then estimate both point estimates
and standard errors according to traditional OLS (Gauss-Markov) and White.

### Input
- `NSim::Int`:    number of simulations (eg. 3000)
- `T::Int`:       sample length (eg. 200)
- `α::Number`:    degree of heteroskedasticity

"""
function SimOLS(NSim,T,α)

    (bLS,StdLS,StdWhite) = [fill(NaN,NSim) for i = 1:3]
    for i = 1:NSim
        f = randn(T)                           #some random regressors
        ϵ = randn(T) .* (1 .+ α*abs.(f))       #heteroskedastic residuals
        y = 1 .+ 0.9*f + ϵ
        x = [f ones(T)]
        (b,u,) = OlsGMFn(y,x)                 #OLS, point estimates
        bLS[i] = b[1]
        Sxx         = x'x
        S           = (x.*u)'*(x.*u)
        V_W         = inv(Sxx)'S*inv(Sxx)      #Cov(b), White
        StdWhite[i] = sqrt(V_W[1,1])
        V_iid       = inv(Sxx)*var(u)          #OLS, traditional
        StdLS[i]    = sqrt(V_iid[1,1])
    end

    return bLS, StdLS, StdWhite

end
##------------------------------------------------------------------------------
