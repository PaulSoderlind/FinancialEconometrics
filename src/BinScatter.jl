"""
    BinScatter(y,x₁,x₂,L=[],U=[],N=20,critval=1.645)

Do a regression `y = x₁'γ + d'β + u`, where `d` is an N-vector indicating
membership in a certain `x₂` bin. Plotting `β` against those bins is a binscatter plot.

### Input:
- `y::Vector`:         dependent variable
- `x₁::VecOrMat`:      control variables
- `x₂::Vector`:        main regressor of interest
- `L::Vector`:         lower bin boundaries, if [] then quantiles (see N)
- `U::Vector`:         upper bin boundaries, if [] then quantiles (see N)
- `N::Vector`:         number of quantiles, giving N+1 bins. Used if `L=U=[]`
- `critval::Vector`:   for calculation of confidence band

### Output
- `β::Vector`:         N-vector of coeffs on the bin (x₂) dummies
- `stdβ::Vector`:      N-vector of std of β
- `fnO::NamedTuple`:   with (LU,confBand)

"""
function BinScatter(y,x₁,x₂,L=[],U=[],N=20,critval=1.645)

  size(x₂,2) > 1 && error("x₂ must be a vector")

  if isempty(L) && isempty(U)
    p     = quantile(x₂,range(0,1,1+N))
    L     = p[1:end-1]
    L[1] -= eps(L[1])     #to make sure minimum(x₂) is in first bin
    U     = p[2:end]
  end
  N = length(L)

  (D,D_colchk,y_binchk) = BinDummies(x₂,L,U)

  if any(==(true),y_binchk) || any(==(true),D_colchk)       #checking if bins are correct
    println("the following x₂ indices are in no bin:")
    printmat(findall(==(true),y_binchk))
    println("the following bins are empty:")
    printmat(findall(==(true),D_colchk))
    error("some x₂ are in no bin or some bins are empty")
  end

  if any(allequal,eachcol(x₁)) && all(==(1),sum(D,dims=2))
    error("don't include a constant in x₁ if the x₂ dummies sum to 1")
  end

  if isempty(x₁)                      #no x₁ controls
    (β,_,_,Covβ,) = OlsNW(y,D)
  else                                #with x₁ controls
    (b,_,_,Covb,) = OlsNW(y,[D x₁])
    β    = b[1:N]                     #coefs on the dummies in D (for x₂)
    Covβ = Covb[1:N,1:N]
    #γ    = b[N+1:end]                #coefs on control variables x₁
  end
  stdβ = sqrt.(diag(Covβ))
  confBand = hcat(β - critval*stdβ,β + critval*stdβ)

  LU = [L U]

  fnO = (;LU,confBand)

  return β, stdβ, fnO

end


"""
  BinDummies(y::Vector,L,U)

Creates TxK BitArray `D` where `D[i,k]=true` if `L[k]<y[i]<=U[k]`

### Input
- `y::Vector`:      T-vector of data
- `L::Vector`:      K-vector of lower bin boundary
- `U::Vector`:      K-vector of upper bin boundary

### Output
- `D::VecOrMat`:       TxK BitArray
- `D_colchk::Vector`:  K-vector, `true` if column `j` in `D` is empty (full of `false`), for checking
- `y_binchk::Vector`:  T-vector, `true` if `y[i]` is in no bin, for checking

"""
function BinDummies(y::Vector,L,U)

  any(L .> U) && error("Must have L <= U")
  (!issorted(L) || !issorted(U)) && error("L and U must be sorted")

  (T,K) = (length(y),length(L))
  _CompareFn(j,yi,L,U) = L[j]<yi<=U[j]

  D = falses(T,K)
  for (i,yi) in enumerate(y)
    k = findfirst(j->_CompareFn(j,yi,L,U),1:K)            #can only be in one bin, findfirst
    !isnothing(k) && (D[i,k] = true)
  end
  (K==1) && (D=vec(D))                                    #to vector if K == 1

  D_colchk = [all(==(false),col) for col in eachcol(D)]   #true if D[:,j] is empty
  y_binchk = vec(sum(D,dims=2) .== 0)                     #true if y[i] is in no group

  return D, D_colchk, y_binchk

end
