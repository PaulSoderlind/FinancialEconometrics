##------------------------------------------------------------------------------
"""
    ARMAFilter(x,rho=[],theta=[],theta0=1.0,z0=[])

Calculate ARMA(p,q) transformation of an input series x. Uses explicit loop (instead of
DSP.filt).


# Input
- `x::Vector`:          T-vector with an input series
- `rho::Vector`:        (optional) p-vector of autoregression coefficients, could be []
- `theta::Vector`:      (optional) q-vector of moving average coefficients (lag 1 to q), could be []
- `theta0::Number`:     (optional) scalar, coefficient on x[t] in MA part, [1]
- `z0::Vector`:         (optional) p-vector, initial values of z, eg. [z(-1);z(0)], []

# Output
- `z::Vector`:          T-vector with output from the filter

# Notice
(a) The process is
    z[t] = rho[1]*z[t-1] + ... + rho[p]*z[t-p] + y[t], where
    y[t] = theta0*x[t]   +  theta[1]*x[t-1] + ... + theta[q]*x[t-q]
(b) The initial values of y are assumed to be zero (or z0)
(c) The (time) order of z0 is as for x
(d) The initial values of x are assumed to be zero
(e) To calculate impulse response functions, use x = [1;zeros(T-1,1)]
(f) There are no initial values in a pure MA and the case of q > p
    is handled with padding with zeros (see the code below)

"""
function ARMAFilter(x::Vector,rho=Float64[],theta=Float64[],theta0=1,z0=Float64[])

  (T,q,p) = (length(x),length(theta),length(rho))
  r       = max(p,q)                         #r obs are needed for start-up

  x  = [zeros(r);x]                          #pad with zeros, clumsy but easy
  Tb = r + T

  theta_ud = q > 0 ? reverse([theta0;theta]) : theta0  #flip upside down
  rho_ud   = p > 1 ? reverse(rho)            : rho

  Tz0 = size(z0,1)
  if isempty(z0)                             #to store output in
    z = zeros(Tb)
  else
    z = [zeros(r-p);z0;zeros(Tb-(r-p)-Tz0)]  #pad with zeros if q > p
  end                                        #zeros(r-p) is empty if r==p

  for t = r+1:Tb
      if p == 0                              #pure MA
        z[t] = dot(theta_ud,x[t-q:t])
      else                                   #AR or ARMA
        z[t] = dot(rho_ud,z[t-p:t-1]) + dot(theta_ud,x[t-q:t])
    end
  end
  z = z[r+1:Tb]                              #cut padded part

  return z

end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
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
    (p != T0) && error("z0 must be $p x $n")

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
##------------------------------------------------------------------------------
