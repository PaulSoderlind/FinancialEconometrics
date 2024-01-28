"""
    IndividualDemean(y,x,id,ϑ=1)

Demean `y` and `x` by individuals (cross-sectional units), assuming `y` is a NT vector
and `x` is NTxK and both are organised according to the NT-vector `id`. The code handles both
the case where data is organised according to id or according to time.

`ϑ=1` is used for the FE estimator, but the GLS estimator uses `0<ϑ<1` (see below).

"""
function IndividualDemean(y,x,id,ϑ=1)

    id_uniq = unique(id)               #which id values are in data set
    N       = length(id_uniq)
    K       = size(x,2)

    (yˣ,xˣ) = (fill(NaN,size(y)),fill(NaN,size(x)))
    (ȳ,x̄)   = (fill(NaN,N),fill(NaN,N,K))
        
    for i = 1:N                              #loop over individuals
        vv_i       = id .== id_uniq[i]           #locate rows which refer to individual i
        ȳ[i]       = mean(y[vv_i])               #averages for individual i
        x̄[i,:]     = mean(x[vv_i,:],dims=1)
        yˣ[vv_i]   = y[vv_i]   .- ϑ*ȳ[i]    
        xˣ[vv_i,:] = x[vv_i,:] .- ϑ*x̄[i:i,:]       #i:i to keep it a row vector
    end

    return yˣ,xˣ,ȳ,x̄

end


"""
    FirstDiff(y,x,id)

Calculate first differences (for each individual) of `y` and `x`,
assuming `y` is a NT vector and `x` is NTxK and both are
organised according to the NT-vector `id`.
It is important, however, that obs (i,t+1) is below that of (i,t).

"""
function FirstDiff(y,x,id)

    id_uniq = unique(id)               #which id values are in data set
    N       = length(id_uniq)
    K       = size(x,2)
  
    (Δy,Δx) = (fill(NaN,size(y)),fill(NaN,size(x)))

    for i = 1:N                          #individual first-differencing, loop over individuals
        vv_i       = id .== id_uniq[i]   ##locate rows which refer to individual i
        Δy[vv_i,:] = y[vv_i]   - lag(y[vv_i,:])   #y[t] -y[t-1]
        Δx[vv_i,:] = x[vv_i,:] - lag(x[vv_i,:])   #x[t,:] -x[t-1,:]        
    end

    (Δy,Δx) = (excise(Δy),excise(Δx))          #cut out rows with NaNs

    return Δy,Δx
    
end    
