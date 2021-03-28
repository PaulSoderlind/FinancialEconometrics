#------------------------------------------------------------------------------
function exciseFn(x)

  z = copy(x)

  vv = vec(any(isnan,z,dims=2))

  if any(vv)              #only keep rows with no NaNs
    vvb = .!vv
    z   = z[vvb,:]
  end

  isa(x,Vector) && (z = z[:,1])   #make z a vector if x is

  return z

end
#------------------------------------------------------------------------------

