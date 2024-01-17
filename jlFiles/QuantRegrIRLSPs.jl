"""
    QuantRegrIRLSPs(y,x,q=0.5;prec=1e-8,epsu=1e-6,maxiter=1000)

Estimate a quantile regression for quantile `q`. The outputs are the point estimates
and three different variance-covariance matrices of the estimates.

### Input
- `y::Vector`:     T vector, dependent variable
- `x::VecOrMat`:   TXK, regressors (including any constant)
- `q::Number`:     quantile to estimate at, 0<q<1
- `prec::Float64`: convergence criterion, 1e-8
- `epsu::Float64`: lower bound on 1/weight, 1e-6
- `maxiter::Int`:  maximum number of iterations, 1000

### Output
- `theta::Vector`: K vector, estimated coefficients
- `vcv::Matrix`:   KxK, traditional covariance matrix
- `vcv2::Matrix`:  KxK, Powell (1991) covariance matrix
- `vcv3::Matrix`:  KxK, Powell (1991) covariance matrix, uniform

### Remarks

1. `while maximum(abs,b - b_old) > prec ...end` creates a loop that continues as long as
    the new and previous estimates differ more than `prec`. However, once the number of
    iterations exceed `maxiter` then the execution is stopped.

2. `u .= max.(u,epsu)` limits how small `u` can become which makes the algorithm
   more stable (recall: the next command is `x./u`).

"""
function QuantRegrIRLSPs(y,x,q=0.5;prec=1e-8,epsu=1e-6,maxiter=1000)

  (T,K) = (size(x,1),size(x,2))
  xw    = copy(x)

  (b_old,b,u,iter) = (zeros(K),fill(1e+6,K) .+ prec,zeros(T),0)

  while maximum(abs,b - b_old) > prec
    copyto!(b_old, b)
    b  .= (xw'*x)\(xw'*y)
    u  .= y - x*b
    #u  .= ifelse.(u.>0,1-q,q).*abs.(u)   #as in Python code, divide loss fn by q(1-q) to get it
    u  .= ifelse.(u.>0,1/q,1/(1-q)).*abs.(u)   #abs(u)/q if u>0, abs(u)/(1-q) if u<0
    u  .= max.(u,epsu)                         #not smaller than epsu
    xw .= x./u
    iter = iter + 1
    if iter > maxiter
      @warn("$iter > maxiter")
      b = NaN
      break
    end
  end

  res = y - x*b

  D   = x'x/T
  h   = 1.06*std(res)/T^0.2                        #Silverman (1986) recommendation
  fx  = exp.(-0.5*((res/h).^2))./(h*sqrt(2*pi))    #Green 7th ed, using N() kernel
  f0  = mean(fx)
  C   = f0*x'x/T
  C_1 = inv(C)
  vcv = q*(1-q)*C_1*D*C_1/T                         #variance-covariance matrix

  C    = (fx.*x)'x/T                                #Wooldrige 2dn ed, Powell 1991
  C_1  = inv(C)                                     #but with Gaussian kernel
  vcv2 = q*(1-q)*C_1*D*C_1/T                        #caputures (x,res) dependence

  fx  = (abs.(res) .<= h/2)/h                       #Wooldrige 2nd ed, Powell 1991
  C   = (fx.*x)'x/T                                 #uniform kernel
  C_1 = inv(C)
  vcv3 = q*(1-q)*C_1*D*C_1/T

  return b, vcv, vcv2, vcv3

end
