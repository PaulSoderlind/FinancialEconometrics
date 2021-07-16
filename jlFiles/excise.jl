#------------------------------------------------------------------------------
"""
    excise(x...)

Remove all lines where the is a NaN/missing in any of the x arrays

# Examples
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

# Input
- `z::Array`: one or several numerical arrays
- `Keepdim::Int`: (keyword) 1 if check rows, 2 if check columns, etc

# Output
- `vvb::BitVector`: vector, element t is true if row (if Keepdim==1) t has no NaN or missing

# Notice
- Set Keepdim=2 if we should instead look for NaNs/missings along rows (and other dimensions).
- For heterogenous arrays like `x=[x1,x1]`, use `FindNNPs(x...)`

Paul.Soderlind@unisg.ch

"""
function FindNNPs(x...;Keepdim=1)

  N  = length(x)
  T     = size(x[1],Keepdim)                 #length of output

  xDims = maximum(ndims,x)                   #max of ndims(x[i]) for i=1:N
  dims  = setdiff(1:xDims,Keepdim)           #dimensions to check

  vvM = falses(T,N)
  for i = 1:N                             #loop over inputs
    vvM[:,i] = any(z -> ismissing(z) || isnan(z),x[i],dims=dims)   #this order is important
  end

  vvb = vec(.!any(vvM,dims=2))      #rows witout NaN/missing in any of the x matrices

  return vvb

end
#------------------------------------------------------------------------------
