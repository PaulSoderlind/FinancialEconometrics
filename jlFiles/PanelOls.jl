#------------------------------------------------------------------------------
"""
    PanelOls(y,x,m=0,clust=[],vvM=[])

Pooled OLS estimation.

# Input
- `y::Matrix`:          TxN matrix with the dependent variable, y(t,i) is for period t, individual i
- `x::3D Array`:        TxKxN matrix with K regressors
- `m::Int`:             (optional), scalar, number of lags in covariance estimation
- `clust::Vector{Int}`: (optional), N vector with cluster number for each individual, [ones(N)]
- `vvM::Matrix`:        (optional), TxN with true/false where false indicates NaN/missings in observation (t,i)

 # Output
 - `fnOutput::NamedTuple`:   named tuple with the following elements
      [1] theta         (K*L)x1 vector, LS estimates of regression coeefficients on kron(z,x)
      [2] CovDK         (K*L)x(K*L) matrix, Driscoll-Kraay covariance matrix
      [3] CovC          covariance matrix, cluster
      [4] CovW          covariance matrix, White's
      [5] R2            scalar, (pseudo-) R2
      [6] yhat          TxN matrix with fitted values
      [7] Nb            T-vector, number of obs in each period

# Notice
- for TxNxK -> TxKxN, do `x = permutedims(z,[1,3,2])`
- for an unbalanced panel, set row t of `(y[t,i],x[t,:,i])` to zeros if there is a NaN/missing value in that row (see vvM)

 Paul.Soderlind@unisg.ch

"""
function PanelOls(y,x,m=0,clust=[],vvM=[])

  (T,N) = (size(y,1),size(y,2))
  K     = size(x,2)                   #TxKxN

  isempty(vvM) && (vvM = trues(T,N))   #handling of NaN/missing
  Nobs = sum(vvM,dims=2)

  isempty(clust) && (clust = ones(Int,N))    #all in the same cluster
  clustU  = unique(clust)             #unique cluster indicators
  G       = length(clustU)            #number of clusters
  vvClust = vec(clust) .== clustU'    #NxG indicators of cluster membership

  xx  = zeros(K,K)                     #Sum[x(t)*x(t)',t=1:T]
  xy  = zeros(K,1)                     #Sum[x(t)*y(t),t=1:T]
  for t = 1:T                             #loop over time
    y_t = y[t,:]                          #dependent variable, Nx1
    x_t = x[t,:,:]'                       #regressors, NxK
    xx .= xx .+ x_t'x_t/N
    xy .= xy .+ x_t'y_t/N
  end

  xx    = xx/T
  xy    = xy/T
  theta = xx\xy                           #ols estimates, solves xx*theta = xy

  yhat = fill(NaN,(T,N))                  #fitted values
  for i = 1:N                             #loop over cross section
    yhat[:,i] = x[:,:,i]*theta
  end
  r = y - yhat                            #fitted residuals

  h = fill(NaN,T,K,N)                     #moment conditions, individual
  for i = 1:N
    h[:,:,i]   = r[:,i].*x[:,:,i]/N
  end
  h_tM = dropdims(sum(h,dims=3),dims=3)   #TxK, moment conditions, aggregated over all

  hG_tM = fill(NaN,T,K,G)                 #moment conditions, aggregated to clusters
  for g = 1:G                             #loop over clusters
    vvg          = vvClust[:,g]
    hG_tM[:,:,g] = sum(h[:,:,vvg],dims=3)
  end

  s2 = sum(abs2,r)/N^2

  (omega0DK,omega0W,omega0C) = [zeros(K,K) for i=1:3]       #DK lag 0,White's,cluster
  (omegajDK,omegajW,omegajC) = [zeros(K,K,m) for i=1:3]     #lags 1 to m
  for t = 1:T                                               #loop over time
    h_it = h[t,:,:]                                         #moment conditions, KxN
    h_t  = h_tM[t,:]                                        #Kx1
    omega0DK .= omega0DK .+ h_t*h_t'                        #Kx1 * 1xK
    omega0W  .= omega0W  .+ h_it*h_it'                      #KxN * NxK
    for g = 1:G                                             #loop over clusters
      omega0C .= omega0C + hG_tM[t,:,g]*hG_tM[t,:,g]'       #Kx1 * 1xK
    end
    for j = 1:min(t-1,m)                                    #0,1,2,2,... for m=2
      omegajDK[:,:,j] = omegajDK[:,:,j] + h_t*h_tM[t-j,:]'  #h(t)*h(t-j)'
      omegajW[:,:,j]  = omegajW[:,:,j]  + h_it*h[t-j,:,:]'
      for g = 1:G                                           #loop over clusters
        omegajC[:,:,j] = omegajC[:,:,j] + hG_tM[t,:,g]*hG_tM[t-j,:,g]'   #Kx1 * 1xK
      end
    end
  end
  #println("")

  ShatDK = NWCovPs(omega0DK,omegajDK,T)          #estimate of S, DK
  ShatC  = NWCovPs(omega0C, omegajC, T)          #estimate of S, cluster
  ShatW  = NWCovPs(omega0W, omegajW, T)          #estimate of S, White's
  s2     = s2/T^2

  xx_1  = inv(xx)
  CovDK = xx_1 * ShatDK  * xx_1'                  #covariance matrix, DK
  CovC  = xx_1 * ShatC   * xx_1'                  #covariance matrix, cluster
  CovW  = xx_1 * ShatW   * xx_1'                  #covariance matrix, White's
  CovLS = xx_1 * s2                               #covariance matrix, LS

  R2  = cor(vec(y[vvM]),vec(yhat[vvM]))^2
  any(.!vvM) && (yhat[.!vvM] .= NaN)              #0 -> NaN

  fnOutput = (;theta,CovDK,CovC,CovW,CovLS,R2,yhat,Nobs)

  return fnOutput

end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    NWCovPs(omega0,omegaj,T)

Calculate covariance matrix of sample average as in Newey-West from a
KxK omega0 matrix and a KxKxm array omegaj. The latter contains m KxK
matrices with autocovariances.
"""
function NWCovPs(omega0,omegaj,T)
  m = size(omegaj,3)
  Shat = omega0/T^2
  for j = 1:m
    Shat .= Shat .+ (1-j/(m+1))*(omegaj[:,:,j] + omegaj[:,:,j]')/T^2
  end
  return Shat
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    FindNNPanel(y,x)

Create TxN matrix with true/false indicating if observation (y[t,i],x[t,:,i])
is valid (that is, contains no NaN/missing)

# Example
- `vv = FindNNPanel(y,x)` where `y` is TxN and `x` is TxKxN

"""
function FindNNPanel(y,x)
  (T,K,N) = size(x)
  vv = falses(T,N)
  for i = 1:N
    vv[:,i] = FindNNPs(view(y,:,i),view(x,:,:,i))
  end
  return vv
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    replaceNaNinYX!(Y,X)

Replaces any rows in Y[:,i] and X[:,:,i] with zeros if there is any NaN/missing.
Can be used to prepare an unbalanced panel data set for use in a fuction meant
for a balanced data set. The function overwrites (Y,X) and also outputs a TxN
matrix vvM indicating with (t,i) observations that have been zeroed out.

# Notice
- The function overwrites (Y,X)
- Recall that `NaN*0 = NaN`, so the code must explicitly set some values to 0
"""
function replaceNaNinYX!(Y,X)

  N = size(Y,2)

  vvM = FindNNPanel(Y,X)              #TxN, rows that have no NaNs/missings

  for i = 1:N                         #loop over cross-section
    vvi         = .!vvM[:,i]          #rows that should be set to 0.0
    Y[vvi,i]   .= 0.0
    X[vvi,:,i] .= 0.0
  end

  return vvM

end
#------------------------------------------------------------------------------
