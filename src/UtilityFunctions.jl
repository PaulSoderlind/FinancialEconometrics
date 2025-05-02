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


"""

    Readcsv(DataFile,SkipCols=1,NaNCode="NA";ToFloat=false,delim=',')


Read a csv file and create a named tuple with variables, with the names
taken from the header.

### Input
- `DataFile::String`:           file path
- `SkipCols::Int`:              number of leading columns to skip (often 1)
- `NaNCode::String or Number`:  indicator for missing value in the file ("NA",-999.99, etc)
- `ToFloat::Bool`:              if true: convert all Ints and Floats to Float64
- `delim::Char`:                typically ',' or '\t'

### Output
- `d::NamedTuple`:              with data, as d.X, d.Y, etc

### Remark
- The conversion of Ints and Floats is to Int64 (with missing) and Float64 (with NaN), respectively,
but with `ToFloat=true` all Ints and Floats are converted to Float64 (wih NaN).


"""
function Readcsv(DataFile,SkipCols=1,NaNCode="NA";ToFloat=false,delim=',')

  (xx,header) = readdlm(DataFile,delim;header=true)
  n = size(xx,2)
  xx = replace(z -> z==NaNCode ? missing : z,xx)
  namesB = tuple(Symbol.(header[SkipCols+1:end])...)       #a tuple (:X,:Y,:Z)

  dx = Vector{Any}(undef,n)
  for i in 1:n
    xx_i = xx[:,i]
    intQ = all(z->isa(z,Union{Integer,Missing}),xx_i)                             #all are integers
    fltQ = any(z->isa(z,AbstractFloat),xx_i) && all(z->isa(z,Union{Real,Missing}),xx_i)  #all are real and some are floats
    misQ = any(ismissing,xx_i)
    #println(i," ",xx_i[1]," ",intQ," ",fltQ," ",misQ)
    if intQ && !ToFloat
      dx[i] = misQ ? convert.(Union{Int,Missing},xx_i) : convert.(Int,xx_i)
    elseif (intQ && ToFloat) || fltQ
      misQ && (xx_i  = replace(z -> ismissing(z) ? NaN : z,xx_i))
      dx[i] = convert.(Float64,xx_i)
    else
      dx[i] = xx_i
    end
  end
  d = NamedTuple{namesB}(dx[SkipCols+1:end])

  return d

end


"""
    lag(x,n=1)

Create a matrix or vector of lagged values.

### Input
- `x::VecOrMat`: T Vector or Txk matrix
- `n::Int`:      scalar, order of lag. For instance, 2 puts x[t-2,:] on row t

### Output
- `z::Array`:  Txk matrix of lags

"""
function lag(x,n=1)

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


"""
    excise(x...;xtype=Any)

Remove all lines where the is a NaN/missing in any of the x arrays

### Input
- `xtype::Type`:      type to convert the data to

### Examples
- `x1 = excise(x)`
- `(y1,x1) = excise(y,x;xtype=Float64)`

"""
function excise(x...;xtype=Any)
  n  = length(x)
  vv = FindNN(x...)         #find rows with NaN/missing
  if xtype != Any
    z = ntuple(i->convert.(xtype,copy(selectdim(x[i],1,vv))),n)    #create a tuple of arrays
  else
    z = ntuple(i->copy(selectdim(x[i],1,vv)),n)    #create a tuple of arrays
  end
  (n==1) && (z = z[1])                           #if a single array in the tuple
  return z
end


"""
    FindNN(x...;Keepdim=1)

Find rows (if Keepdim==1) which have no NaNs missing in other dimensions (eg. in no columns).

### Input
- `z::Array`: one or several numerical arrays
- `Keepdim::Int`: (keyword) 1 if check rows, 2 if check columns, etc

### Output
- `vvb::BitVector`: vector, element t is true if row (if Keepdim==1) t has no NaN or missing

### Notice
- Set Keepdim=2 if we should instead look for NaNs/missings along rows (and other dimensions).
- For heterogenous arrays like `x=[x1,x1]`, use `FindNN(x...)`

Paul.Soderlind@unisg.ch

"""
function FindNN(x...;Keepdim=1)

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


"""
    CovToCor(covMat)

Compute correlation matrix from covariance matrix.

"""
function CovToCor(covMat)
  d      = diag(covMat)            #variances
  corMat = covMat./sqrt.(d*d')
  return corMat
end


"""
    logpdfNorm(x)

Calculate the log pdf of a N(0,1) variable at x.
Should give the same as `Distributions.logpdf(Normal(0,1),x)`

"""
function logpdfNorm(x)
    f = -0.5*log(2*π) - 0.5*x^2
    return f
  end

