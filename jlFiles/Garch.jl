#------------------------------------------------------------------------------
function egarch11LL(par::Vector,y,x)

  (T,k) = (size(x,1),size(x,2))

  #lns2(t) = omega + alpha*abs(u(t-1)/s[t-1]) + beta1*lns2(t-1) + gamma1*u(t-1)/s[t-1]
  b = par[1:k]                                     #mean equation, y = x'*b
  (omega,alpha,beta1,gamma1) = par[k+1:k+4]
  yhat = x*b
  u    = y - yhat
  s2_0 = var(u)

  lns2    = zeros(typeof(alpha),T)
  lns2[1] = log(s2_0)
  for t = 2:T
    s_t1   = sqrt(exp(lns2[t-1]))
    lns2[t] = omega + alpha*abs(u[t-1])/s_t1 + beta1*lns2[t-1] + gamma1*u[t-1]/s_t1
  end
  s2 = exp.(lns2)

  LL_t    = -(1/2)*log(2*pi) .- (1/2)*log.(s2) - (1/2)*(u.^2)./s2
  LL_t[1] = 0

  LL = sum(LL_t)

  return LL,LL_t,s2,yhat,u

end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

function DccLL(par,v,s2,Qbar)

  #(α,β) = par
  (α,β) = DccParTrans(par)             #restrict

  (T,n) = (size(v,1),size(v,2))

  u = v .* sqrt.(s2)

  Sigma = fill(NaN,(n,n,T))
  Q_t   = copy(Qbar)                    #starting guess
  LL_t  = zeros(T)
  for t = 2:T
    Q_t     = (1-α-β)*Qbar + α*v[t-1,:]*v[t-1,:]' + β*Q_t
    q_t     = diag(Q_t)
    R_t     = Q_t./sqrt.(q_t*q_t')
    d_t     = s2[t,:]
    Sigma_t = R_t.*sqrt.(d_t*d_t')
    LL_t[t] = -n*log(2*pi) - logdet(Sigma_t) - u[t,:]'*inv(Sigma_t)*u[t,:]
    Sigma[:,:,t] = Sigma_t
  end

  LL_t = 0.5*LL_t
  LL = sum(LL_t)

  return LL, LL_t, Sigma

end
#------------------------------------------------------------------------------

function DccParTrans(par)
    (a,b) = par
    α      = exp(a)/(1+exp(a)+exp(b))
    β      = exp(b)/(1+exp(a)+exp(b))
  return α,β
end
#------------------------------------------------------------------------------
