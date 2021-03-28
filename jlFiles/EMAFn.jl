"""
    EMAFn(x,m=1)

Equally weighted moving average (over the current and m-1 lagged values) for a matrix `x`.

"""
function EMAFn(x,m=1)

    (T,n) = (size(x,1),size(x,2))
    q = m - 1                               #m=2, q=1 (avg over [t-1,t])

    (q < 0) && error("q must be >= 0")

    y = fill(NaN,T,n)                       #y[t] = (x[t-1] + x[t])/2
    for t = 1:T
        vv     = max(1,t-q):t                 #with q=2; 1:1,1:2,1:3,2:4
        y[t,:] = sum(x[vv,:],dims=1)/m
    end

    return y

end
