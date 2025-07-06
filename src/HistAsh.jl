"""
    HistcNorm(x,edges)

Create normalized histogram (area=1) and reports the mid point of the bins


# Input
- `x::Vector`:         data
- `edges::Vector`:     range, K+1 edges, for K right closed intervals,
  counted if edges[k] < x[i] <= edges[k+1]

# Output
- `xProb::Vector`:    K-vector
- `xPdf::Vector`:     K-vector, scaled frequencies for the K bins, (histogram integrates to 1)
- `BinMid::Vector`:   mid points of the intervals, eg. [1.5,2.5,3.5,4.5]

# Remark
- NaNs are ignored in the computations
- Should give the same result as `h = fit(Histogram,x,edges,closed=:right)`

"""
function HistcNorm(x,edges)

  h      = diff(edges)
  BinMid = edges[1:end-1] + h/2

  K = length(edges) - 1                 #number of intervals
  T = sum(!isnan,x)                     #count only non-NaN data
  xProb = zeros(K)
  for k in 1:K
    (lo,hi)  = edges[k:k+1]
    xProb[k] = count(z->(lo<z<=hi),x)/T
  end

  xPdf = xProb./h                                          #to make histogram integrate to unit

  return xProb,xPdf,BinMid

end


"""
    HistAsh(x,edges,m=1)

Calculate averaged shifted histogram

# Input
- `x::Vector`:              T-vector, a data series
- `edges::AbstractRange`:   right closed intervals, counted if edges[k] < x[i] <= edges[k+1]
- `m::Int`:                 number of different shifts

# Output
- `ashNorm::Vector`:        averaged shifted histogram values normalised to have an area of 1
- `BinMid::Vector`:        mid points of (all the smaller) intervals
- `edgesM::AbstractRange`:  edges in the shifted histogram

# Examples
```
x     = [3.4,3.9,4.9]          #from lecture notes
edges = 1:6
m     = 2
```

# Remark
- NaNs are ignored in the computations

"""
function HistAsh(x,edges::AbstractRange,m=1)

  T = sum(!isnan,x)
  h = step(edges)
  δ = h/m

  edgesM = range(edges[1]-(m-1)*δ,edges[end]+(m-1)*δ,step=δ)   #edges of smaller bins
  NBins  = length(edgesM) - 1                                  #no. smaller bins

  (xProb,_,BinMidM) = HistcNorm(x,edgesM)
  nM               = T*xProb                             #no. obs in each small bin

  #!isapprox(sum(nM),T) && error("some data is outside the bin grid")

  i = (1-m):(m-1)               #lags and leads
  w = m .- abs.(i)              #weights: 1,2,3,2,1 for m = 3

  ash = zeros(NBins)
  for j in m:(NBins-m+1)        #3,4,..,23 if m = 3 and NBins = 25
    vv     = j-m+1:j+m-1
    ash[j] = dot(w,nM[vv])
    #println("$j $vv $(ash[j])")
  end
  ashN    = ash/m                #average
  ashNorm = ash/(m*T*h)          #normalised as a pdf

  return ashNorm, BinMidM, edgesM

end
