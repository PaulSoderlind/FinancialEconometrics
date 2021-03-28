"""
    lagFn(x,n=1)

Create a matrix or vector of lagged values.

# Input
- `x::Array`: T Vector or Txk matrix
- `n::Int`:   scalar, order of lag. For instance, 2 puts x[t-2,:] on row t

# Output
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
