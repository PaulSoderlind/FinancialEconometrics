##------------------------------------------------------------------------------
"""
    ReturnStats(Re,Annfactor=252)

Calculate average excess return, the std and the SR ratio - and annualise.
Returns a 3xn matrix with (annualised) `[μ;σ;SR]`, where `n=size(Re,2)`.
"""
function ReturnStats(Re,Annfactor=252)
    μ  = mean(Re,dims=1)*Annfactor
    σ  = std(Re,dims=1)*sqrt(Annfactor)
    SR = μ./σ
    stats = [μ;σ;SR]
    return stats
end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    PutDataInNT(x,header)

Creates a NamedTuple with, for instance, `N.X`, `N.Y` and `N.Z` where `x` is a matrix
and `header = ["X" "Y" "Z"]` or `["X","Y","Z"]`.

"""
function PutDataInNT(x,header)
    namesB = tuple(Symbol.(header)...)                            #a tuple (:X,:Y,:Z)
    N      = NamedTuple{namesB}([x[:,i] for i=1:size(x,2)])       #NamedTuple with N.X, N.Y and N.Z
    return N
end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    lagFn(x,n=1)

Create a matrix or vector of lagged values.

### Input
- `x::Array`: T Vector or Txk matrix
- `n::Int`:   scalar, order of lag. For instance, 2 puts x[t-2,:] on row t

### Output
- `z::Array`:  Txk matrix of lags

"""
function lagFn(x,n=1)

  (T,k) = (size(x,1),size(x,2))

  if n < 0                                    #leads
    z = [ x[1-n:T,:]; fill(NaN,(abs(n),k)) ]
  elseif n == 0                               #no lag or lead
    z = copy(x)
  elseif n > 0                                #lags
    z = [ fill(NaN,(n,k)); x[1:T-n,:] ]
  end

  isa(x,AbstractVector) && (z = z[:,1])               #z should be vector if x is

  return z

end
##------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    excise(x...)

Remove all lines where the is a NaN/missing in any of the x arrays

### Examples
- `x1 = excise(x)`
- `(y1,x1) = excise(y,x)`

"""
function excise(x...)
  n  = length(x)
  vv = FindNNPs(x...)         #find rows with NaN/missing
  z = ntuple(i->copy(selectdim(x[i],1,vv)),n)    #create a tuple of arrays
  (n==1) && (z = z[1])                           #if a single array in the tuple
  return z
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    FindNNPs(x...;Keepdim=1)

Find rows (if Keepdim==1) which have no NaNs missing in other dimensions (eg. in no columns).

### Input
- `z::Array`: one or several numerical arrays
- `Keepdim::Int`: (keyword) 1 if check rows, 2 if check columns, etc

### Output
- `vvb::BitVector`: vector, element t is true if row (if Keepdim==1) t has no NaN or missing

### Notice
- Set Keepdim=2 if we should instead look for NaNs/missings along rows (and other dimensions).
- For heterogenous arrays like `x=[x1,x1]`, use `FindNNPs(x...)`

Paul.Soderlind@unisg.ch

"""
function FindNNPs(x...;Keepdim=1)

  N  = length(x)
  T  = size(x[1],Keepdim)                    #length of output

  xDims = maximum(ndims,x)                   #max of ndims(x[i]) for i=1:N
  dims  = setdiff(1:xDims,Keepdim)           #dimensions to check

  vvM = falses(T,N)
  for i = 1:N                             #loop over inputs
    vvM[:,i] = any(isunordered,x[i],dims=dims)
  end

  vvb = vec(.!any(vvM,dims=2))      #rows witout NaN/missing in any of the x matrices

  return vvb

end
#------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""

    OLSyxReplaceNaN(Y,X)

Replaces any rows in Y and X with zeros if there is any NaN/missing in any of them.

"""
function OLSyxReplaceNaN(Y,X)

  vv = FindNNPs(Y,X)             #vv[t] = true if no missing/NaN i (y[t],x[t,:])

  (Yb,Xb)     = (copy(Y),copy(X))    #set both y[t] and x[t,:] to 0 if any missing/NaN for obs. t
  Yb[.!vv]   .=  0
  Xb[.!vv,:] .= 0

  return vv, Yb, Xb

end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    EMAFn(x,m=1)

Equally weighted moving average (over the current and m-1 lagged values) of each column
in a matrix `x`.

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
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    CovToCor(covMat)

Compute correlation matrix from covariance matrix.

"""
function CovToCor(covMat)
  d      = diag(covMat)            #variances
  corMat = covMat./sqrt.(d*d')
  return corMat
end
##------------------------------------------------------------------------------
