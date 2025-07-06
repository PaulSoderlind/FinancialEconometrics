"""
    EWMA_variance(x,λ,μ_start=NaN,σ²_start=NaN)

Calculate Txn matrices of EWMA means and variances from n series

"""
function EWMA_variance(x,λ,μ_start=NaN,σ²_start=NaN)
    T = length(x)
    (μ,σ²) = (fill(μ_start,T),fill(σ²_start,T))
    for t in 2:T
        μ[t]    = λ*μ[t-1]  + (1-λ)*x[t-1]
        σ²[t]   = λ*σ²[t-1] + (1-λ)*(x[t-1]-μ[t-1])^2
    end
    return μ,σ²
  end


"""
    EWMA_covariance(v,λ,Q0=NaN)

Calculate Txnxn array of T EWMA covariance matrices of v
"""
function EWMA_covariance(v,λ,Q0=NaN)
  isnan(Q0) && (Q0 = cov(v))
  (T,n) = (size(v,1),size(v,2))
  Q = fill(NaN,T,n,n)
  Q[1,:,:] = Q0
  for t in 2:T
      Q[t,:,:] = λ*Q[t-1,:,:] + (1-λ)*v[t-1,:]*v[t-1,:]'
  end
  return Q
end


"""
    Dcc_EWMA(x,λ)

Do EWMA version of DCC
"""
function Dcc_EWMA(x,λ)

  (T,n) = (size(x,1),size(x,2))

  (μ,σ²) = (fill(NaN,T,n),fill(NaN,T,n))
  for i in 1:n
    x_i = x[:,i]
    (μ[:,i],σ²[:,i]) =  EWMA_variance(x_i,λ,mean(x_i),var(x_i))
  end

  v = (x-μ)./sqrt.(σ²)                #standardised residuals
  Q = EWMA_covariance(v,λ)

  (R,S) = (similar(Q),similar(Q))
  for t in 1:T
    Q_t = Q[t,:,:]
    q_t = diag(Q_t)
    R[t,:,:] = Q_t./sqrt.(q_t*q_t')       #to a correlation matrix
    S[t,:,:] = R[t,:,:].*sqrt.(σ²[t,:]*σ²[t,:]')   #to covariance matrix
  end

  return S,R,μ,σ²

end
