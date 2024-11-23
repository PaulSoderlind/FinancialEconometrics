"""
    PanelOls(y,x,m=0,clust=[];FixNaNQ=false)

Pooled OLS estimation.

### Input
- `y0::Matrix`:         TxN matrix with the dependent variable, `y[t,i]` is for period `t`, individual `i`
- `x0::3D Array`:       TxKxN matrix with K regressors
- `m::Int`:             (optional), scalar, number of lags in covariance estimation
- `clust::Vector{Int}`: (optional), N vector with cluster number for each individual, [`ones(N)`]
- `FixNaNQ::Bool`:      (optional), true: replace all cases (y[t,i],x[t,:,i]) with some NaN/missing with (0,0), using PanelReplaceNaN()

 ### Output
 - `fnOutput::NamedTuple`:   named tuple with the following elements
      1. `theta`         (K*L)x1 vector, LS estimates of regression coeefficients on kron(z,x)
      2. `CovDK`         (K*L)x(K*L) matrix, Driscoll-Kraay covariance matrix
      3. `CovC`          covariance matrix, cluster
      4. `CovNW`         covariance matrix, Newey-West (or White if m=0)
      5. `CovLS`         covariance matrix, iid
      6. `R2`            scalar, (pseudo-) R2
      7. `yhat`          TxN matrix with fitted values
      8. `Nb`            T-vector, number of obs in each period

### Notice
- for TxNxK -> TxKxN, do `x = permutedims(z,[1,3,2])`

"""
function PanelOls(y0,x0,m=0,clust=[];FixNaNQ=false)

  (T,N) = (size(y0,1),size(y0,2))
  K     = size(x0,2)                   #TxKxN

  if FixNaNQ
    (vvM,y,x) = PanelReplaceNaN(y0,x0)
  else
    vvM   = FindNNPanel(y0,x0)
    any(!,vvM) && error("still some NaN/missing")
    (y,x) = (y0,x0)
  end

  Nb  = vec(sum(vvM,dims=2))           #observations in each period
  Tb  = sum(>(0),Nb)                   #effective number of time periods
  TNb = sum(vvM)                       #effective number of observations

  isempty(clust) && (clust = ones(Int,N))    #all in the same cluster
  clustU  = unique(clust)             #unique cluster indicators
  G       = length(clustU)            #number of clusters
  vvClust = vec(clust) .== clustU'    #NxG indicators of cluster membership

  xx  = zeros(K,K)                     #Sum[x(t)*x(t)',t=1:T]
  xy  = zeros(K)                       #Sum[x(t)*y(t),t=1:T]
  for t in 1:T                            #loop over time
    y_t = y[t,:]                          #dependent variable, Nx1
    x_t = x[t,:,:]'                       #regressors, NxK
    xx .= xx .+ x_t'x_t
    xy .= xy .+ x_t'y_t
  end

  xx    = xx
  xy    = xy
  theta = xx\xy                           #ols estimates, solves xx*theta = xy

  yhat = fill(NaN,T,N)                    #fitted values
  for i in 1:N                            #loop over cross section
    yhat[:,i] = x[:,:,i]*theta
  end
  r = y - yhat                            #fitted residuals

  h = fill(NaN,T,K,N)                     #moment conditions, individual
  for i in 1:N
    h[:,:,i]   = r[:,i].*x[:,:,i]
  end
  h_tM = dropdims(sum(h,dims=3),dims=3)   #TxK, moment conditions, aggregated over all

  hG_tM = fill(NaN,T,K,G)                 #moment conditions, aggregated to clusters
  for g in 1:G                            #loop over clusters
    vvg          = vvClust[:,g]
    hG_tM[:,:,g] = sum(view(h,:,:,vvg),dims=3)
  end

  s2 = sum(abs2,r)/TNb

  (omega0DK,omega0W,omega0C) = [zeros(K,K) for i in 1:3]    #DK lag 0,White's,cluster
  (omegajDK,omegajW,omegajC) = [zeros(K,K,m) for i in 1:3]  #lags 1 to m
  for t in 1:T                                              #loop over time
    h_it = h[t,:,:]                                         #moment conditions, KxN
    h_t  = h_tM[t,:]                                        #Kx1
    omega0DK .= omega0DK .+ h_t*h_t'                        #Kx1 * 1xK
    omega0W  .= omega0W  .+ h_it*h_it'                      #KxN * NxK
    for g in 1:G                                            #loop over clusters
      omega0C .= omega0C + hG_tM[t,:,g]*hG_tM[t,:,g]'       #Kx1 * 1xK
    end
    for j in 1:min(t-1,m)                                    #0,1,2,2,... for m=2
      omegajDK[:,:,j] = omegajDK[:,:,j] + h_t*h_tM[t-j,:]'  #h(t)*h(t-j)'
      omegajW[:,:,j]  = omegajW[:,:,j]  + h_it*h[t-j,:,:]'
      for g in 1:G                                           #loop over clusters
        omegajC[:,:,j] = omegajC[:,:,j] + hG_tM[t,:,g]*hG_tM[t-j,:,g]'   #Kx1 * 1xK
      end
    end
  end

  ShatDK = NWCovPs(omega0DK,omegajDK,1)          #estimate of S, DK
  ShatC  = NWCovPs(omega0C, omegajC, 1)          #estimate of S, cluster
  ShatW  = NWCovPs(omega0W, omegajW, 1)          #estimate of S, White's

  xx_1  = inv(xx)
  CovDK = xx_1 * ShatDK  * xx_1'                  #covariance matrix, DK
  CovC  = xx_1 * ShatC   * xx_1'                  #covariance matrix, cluster
  CovNW = xx_1 * ShatW   * xx_1'                  #covariance matrix, Newey-West (White if m=0)
  CovLS = xx_1 * s2                               #covariance matrix, LS

  R2  = cor(vec(y[vvM]),vec(yhat[vvM]))^2
  any(.!vvM) && (yhat[.!vvM] .= NaN)              #0 -> NaN

  fnOutput = (;theta,CovDK,CovC,CovNW,CovLS,R2,yhat,Nb)

  return fnOutput

end


"""
    NWCovPs(omega0,omegaj,T)

Calculate covariance matrix of sample average as in Newey-West from a
KxK `omega0` matrix and a KxKxm array `omegaj`. The latter contains m KxK
matrices with autocovariances.
"""
function NWCovPs(omega0,omegaj,T)
  m = size(omegaj,3)
  Shat = omega0/T^2
  for j in 1:m
    Shat .= Shat .+ (1-j/(m+1))*(omegaj[:,:,j] + omegaj[:,:,j]')/T^2
  end
  return Shat
end


"""
    PanelReshuffle(y,x,z,id,per)

Reshuffle the dependent variable into an TxN matrix `Y` (`T` periods, `N` cross-sectional units)
and the regressors into a TxKxN array `X`. Handles unbalanced data sets.
This reshuffling allows the `PaneslOls()` function to
handle autocorrelation and cross-sectional clustering.

### Input
- `y::Vector`:       NT-vector of dependent variable, NT <= N*T
- `x::VecOrMat`:     NTxK matrix with regressors
- `z::VecOrMat`:     NTxL matrix with dummies (or other variables used for fixed effects, perhaps `[]`)
- `id::Vector`:      NT-vector of identity of cross-sectional unit
- `per::Vector`:     NT-vector of period (time) indicator

### Output
- `t::Tuple`:         (`Y,X`) TxN and TxKxN if `z=[]`, otherwise (`Y,X,Z`) where `Z` is TxLxN

"""
function PanelReshuffle(y,x,z,id,per)

    id_uniq   = sort(unique(id))
    N         = length(id_uniq)      #no. cross-sectional units
    per_uniq  = sort(unique(per))
    T         = length(per_uniq)     #no. periods
    K         = size(x,2)            #no. regressors

    Y = fill(NaN,T,N)                #TxN
    X = fill(NaN,T,K,N)              #TxKxN
    Z = fill(NaN,T,size(z,2),N)
    for i in 1:N
        vv_i  = id .== id_uniq[i]   #rows in y which refer to individual i
        per_i = per[vv_i]           #periods for which individual i has data
        vv_t  = indexin(per_i,per_uniq)
        Y[vv_t,i]   = y[vv_i]
        X[vv_t,:,i] = x[vv_i,:]
        !isempty(z) && (Z[vv_t,:,i] = z[vv_i,:])
    end

    t = isempty(z) ? (Y,X) : (Y,X,Z)
    return t

end


"""
    FindNNPanel(y,x,z=[])

Create TxN matrix with true/false indicating if observation `(y[t,i],x[t,:,i])`
is valid (that is, contains no NaN/missing)

### Example
- `vv = FindNNPanel(y,x)` where `y` is TxN and `x` is TxKxN

### Requires
- the function `FindNN()`

"""
function FindNNPanel(y,x,z=[])
  (T,K,N) = size(x)
  vv = falses(T,N)
  for i = 1:N
    if isempty(z)
      vv[:,i] = FindNN(view(y,:,i),view(x,:,:,i))
    else
      vv[:,i] = FindNN(view(y,:,i),view(x,:,:,i),view(z,:,:,i))
    end
  end
  return vv
end


"""
    PanelReplaceNaN(Y,X,Z=[])

Replaces any rows in `Y[:,i]` and `X[:,:,i]` (and optionally `Z[:,:,i]`) with zeros
if there is any NaN/missing. Can be used to prepare an unbalanced panel data set for
use in a fuction meant for a balanced data set. The function creates `(Yb,Xb)` or `(Yb,Xb,Zb)`
and also outputs a TxN matrix `vvM` indicating with (t,i) observations that have been zeroed out.

### Requires
- the function `FindNNPanel()`

### Notice
- For a similar function that overwrites the existing (Y,X,Z), see PanelReplaceNaN!(Y,X).

"""
function PanelReplaceNaN(Y,X,Z=Float64[])

  (Yb,Xb,Zb) = (copy(Y),copy(X),copy(Z))
  N   = size(Y,2)
  vvM = isempty(Z) ? FindNNPanel(Y,X) : FindNNPanel(Y,X,Z)   #TxN, rows that have no NaNs/missings

  for i in 1:N                        #loop over cross-section
    vvi          = .!vvM[:,i]         #rows that should be set to 0.0
    Yb[vvi,i]   .= 0.0
    Xb[vvi,:,i] .= 0.0
    !isempty(Z) && (Zb[vvi,:,i] .= 0.0)
  end

  t = isempty(Z) ? (vvM,Yb,Xb) : (vvM,Yb,Xb,Zb)

  return t

end


function PanelReplaceNaN!(Y,X,Z=Float64[])

  N   = size(Y,2)
  vvM = isempty(Z) ? FindNNPanel(Y,X) : FindNNPanel(Y,X,Z)   #TxN, rows that have no NaNs/missings

  for i in 1:N                       #loop over cross-section
    vvi         = .!vvM[:,i]         #rows that should be set to 0.0
    Y[vvi,i]   .= 0.0
    X[vvi,:,i] .= 0.0
    !isempty(Z) && (Z[vvi,:,i] .= 0.0)
  end

  return vvM

end


"""
    DummiesCreate(z)

Create TxM dummies from a T-vector of M unique values

"""
function DummiesCreate(z)
    z_uniq = sort(unique(z))
    dum    = z .== permutedims(z_uniq)
    return dum
end


"""
    TimeDummiesTTN(T,N)

Create TxTxN array of time dummies

"""
function TimeDummiesTTN(T,N,drop1Q=true)
    Z = repeat(1.0I[1:T,1:T],outer=(1,1,N))
    drop1Q && (Z = Z[:,2:end,:])
    return Z
end


"""
    IndivDummiesTNN(T,N)

Create TxNxN array of individual dummies

"""
function IndivDummiesTNN(T,N)
    Z = zeros(T,N,N)
    for i in 1:N
        Z[:,i,i] .= 1
    end
    return Z
end


"""
    FixedIndivEffects(y0,x0,z0=[])

Demeans `y0` and `x0` (and possibly any `z0`) by taking out individual and/or time fixed effects.
Loops to handle unbalanced panels. Notice that if `(y[t,i],x[t,:,i])` contains
any missing value/NaN, then this oservation is excluded from the computations.


### Input
- `y0::Matrix`:      TxN matrix of dependent variables
- `x0::Array`:       TxKxN array with regressors, nregressor k for cross-sectional unit i is in `x0[:,k,i]`
- `z0::Array`:       (optional) TxLxN array with dummies etc

### Output
- `y::Matrix`:
- `x::Array`:
- `z::Array`:
- `yxAvg::Named tuple`: `yAvg_i,xAvg_i,zAvg_i,vvt`

### Requires
- the function `FindNNPanel()`

### Notice
- for TxKxN -> TxNxK (or vice versa), do `permutedims(z,[1,3,2])`

"""
function FixedIndivEffects(y0,x0,z0=Float64[])

  (T,N) = (size(y0,1),size(y0,2))
  (K,L) = (size(x0,2),size(z0,2))            #TxKxN

  DoZQ = !isempty(z0)

  vvt = DoZQ ? FindNNPanel(y0,x0,z0) : FindNNPanel(y0,x0)   #use (t,i) data or not

  (yAvg_i,xAvg_i,zAvg_i) = (fill(NaN,1,N),fill(NaN,1,K,N),fill(NaN,1,L,N))
  for i = 1:N              #loop over cross section
    vv            = vvt[:,i]
    yAvg_i[1,i]   = mean(view(y0,vv,i))
    xAvg_i[1,:,i] = mean(view(x0,vv,:,i),dims=1)
    DoZQ && (zAvg_i[1,:,i] = mean(view(z0,vv,:,i),dims=1))
  end
  y = y0 .- yAvg_i                     #subtract fixed individual effects
  x = x0 .- xAvg_i
  z = DoZQ ? z0 .- zAvg_i : Float64[]

  yxAvg = (;yAvg_i,xAvg_i,zAvg_i,vvt)

  return y, x, z, yxAvg

end


"""
    FixedTimeEffects(y0,x0,z0)

Demeans `y0` and `x0` (and possibly any `z0`) by taking out time fixed effects.
Loops to handle unbalanced panels. Notice that if `(y[t,i],x[t,:,i],z[t,:,i])` contains
any missing value/NaN, then this observation is excluded from the computations.


### Input
- `y0::Matrix`:      TxN matrix of dependent variables
- `x0::Array`:       TxKxN array with regressors, nregressor k for cross-sectional unit i is in `x0[:,k,i]`
- `z0::Array`:       (optional) TxLxN array with dummies etc

### Output
- `y::Matrix`:
- `x::Array`:
- `z::Array`:
- `yxAvg::Named tuple`: `yAvg_t,xAvg_t,zAvg_t,vvt`

### Requires
- the function `FindNNPanel()`

"""
function FixedTimeEffects(y0,x0,z0=Float64[])

  (T,N) = (size(y0,1),size(y0,2))
  (K,L) = (size(x0,2),size(z0,2))            #TxKxN

  DoZQ = !isempty(z0)

  vvt = DoZQ ? FindNNPanel(y0,x0,z0) : FindNNPanel(y0,x0)   #use (t,i) data or not

  (yAvg_t,xAvg_t,zAvg_t) = (fill(NaN,T),fill(NaN,T,K),fill(NaN,T,L))
  for t = 1:T              #loop over time
    vv          = vvt[t,:]
    yAvg_t[t]   = mean(view(y0,t,vv))
    xAvg_t[t,:] = mean(view(x0,t,:,vv),dims=2)
    DoZQ && (zAvg_t[t,:] = mean(view(z0,t,:,vv),dims=2))
  end
  y = y0 .- yAvg_t   #subtract individual and time fixed effects
  x = x0 .- xAvg_t
  z = DoZQ ? z0 .- zAvg_t : Float64[]

  yxAvg = (;yAvg_t,xAvg_t,zAvg_t,vvt)

  return y, x, z, yxAvg

end


"""
    FixedIndivTimeEffects(Y,X,p=10)

Estimate 2-way (individual and time) fixed effects model, by first applying an individual
within-adjustment to (Y,X,time dummies), then applying a repeated Frisch-Waugh approach to
partial out the time dummies.  To be used when N > T.

"""
function FixedIndivTimeEffects(Y,X,p=10)

    (T,N)   = size(Y)

    Z       = TimeDummiesTTN(T,N)         #create Tx(T-1)xN time dummies
    (Y,X,Z) = FixedIndivEffects(Y,X,Z)
    vvM     = PanelReplaceNaN!(Y,X,Z)

    (Y,X,Z) = FWonZRepeated!(Y,X,Z,p)

    return Y,X

end



"""
    FixedTimeIndivEffects(Y,X,p=10)

Estimate 2-way (time and individual) fixed effects model, by first applying a time
within-adjustment to (Y,X,individual dummies), then applying a repeated Frisch-Waugh a
pproach to partial out the individual dummies. To be used when T > N.

"""
function FixedTimeIndivEffects(Y,X,p=10)

    (T,N)   = size(Y)

    Z       = IndivDummiesTNN(T,N)            #create TxXxN individual dummies
    (Y,X,Z) = FixedTimeEffects(Y,X,Z)
     vvM    = PanelReplaceNaN!(Y,X,Z)

    (Y,X,Z) = FWonZRepeated!(Y,X,Z,p)

    return Y,X

end


"""
    OlsWrite!(y,x,Sxx_f,InOutMat,i,d1Q=false)

Regress `y` on `x` using a precalculated `Sxx=factorize(x'x)` as input.
The residuals are written to `InOutMat` if `d1Q=true` and to `InOutMat[:,i,:]` if
`d1Q=false`.

"""
function OlsWrite!(y,x,Sxx_f,InOutMat,i,d1Q=false)
    Sxy = x'*y
    b   = Sxx_f\Sxy                  #cf. b = x\y
    if d1Q
      copyto!(InOutMat,y - x*b)      #same as Y[:] = y - x*b
    else
      InOutMat[:,i,:] = y - x*b
    end
    return InOutMat
end


"""
    FWonZRepeated!(Y,X,Z,p=10)

Recursive Frisch-Waugh of (a) (Y,X,Z[:,4:end,]) on Z[:,1:3,:];
(b) get the (Y,X) residuals and then regress those on the Z[:,4:6,:] residuals, etc. Iterate
until we have projected on all columns in Z.

### Input
- `Y::Matrix`:     TxN
- `X::Array`:      TxKxN
- `Z::Array`:      TxLxN
- `p::Int`:        number of Z variables to project on in each iteration

### Output
similar matrices but filled with the residuals after projecting on Z


### Remark
The following could potentially be used (in a future version) to have a tailor made loop.
```
jseq   = vcat(1:p-1,p:p:M)
n_jseq = length(jseq)
for j in 1:n_jseq
  vj = j < n_jseq ? (jseq[j]:jseq[j+1]-1) : (jseq[j]:jseq[end])
end
```

"""
function FWonZRepeated!(Y,X,Z,p=10)

  Zreshape(Z) = reshape( permutedims(Z,(1,3,2)),:,size(Z,2) )     #TxKxN => TNxK matrix

  M = size(Z,2)
  (T,N,K) = (size(Y,1),size(Y,2),size(X,2))

  for j in 1:p:M                                   #loop over 1,4,7, etc for p = 3
    vj = j:min(j+p-1,M)                            #Z[:,vj,:], 1:3,4:6,7:9, etc
    (mod(j-1,p*10)==0) && print("\rProcessed $(vj[1])-$(vj[end]) of $M")
    #println(j," ",vj," ",maximum(vj)+1," ",M)
    Zj    = copy(Zreshape(Z[:,vj,:]))
    Sxx_f = factorize(Zj'*Zj)
    OlsWrite!( vec(Y),Zj,Sxx_f,Y,0,true )                     #Y on Z[:,vj,:]
    Threads.@threads for i in 1:K                             #X[:,i,:] on Z[:,vj,:]
      OlsWrite!( vec(X[:,i,:]),Zj,Sxx_f,X,i,false )
    end
    Threads.@threads for i in maximum(vj)+1:M                 #Z[:,vj+1,:] on Z[:,vj,:]
      OlsWrite!( vec(Z[:,i,:]),Zj,Sxx_f,Z,i,false )
    end
    Z[:,vj,:] .= NaN                                          #done Z[:,vj,:], set to NaN
  end
  println()

  return Y,X,Z

end
