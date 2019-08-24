"""
    VARFilter(x,A,z0)

Create z Txn matrix from VAR model where z[t,:] = A1*z[t-1,:] +...+Ap*z[t-p,:]+ x[t,:].
A is an nxnxp array with cat(A1,A2,...,dims=3)
z0 is pxn initial values of z (for [t=-2;t=-1;t=0] for a VAR(3))
"""
function VARFilter(x,A,z0)
    p     = size(A,3)            #lag order
    (T,n) = (size(x,1),size(x,2))
    T0    = size(z0,1)
    (p !== T0) && error("z0 must be $p x $n")
    z     = [z0;zeros(T,n)]
    x     = [zeros(p,n);x]
    for t = p+1:size(z,1)        #loop over t
        for s = 1:p              #loop over lags, z[t-1],z[t-2],...,z[t-p]
            z[t,:] = z[t,:] + A[:,:,s]*z[t-s,:]  #works since z[t,:] is a vector
        end
        z[t,:] = z[t,:] + x[t,:]
    end
    #printmat(z)
    z = z[p+1:end,:]                #cut padded part
    return z
end
