#------------------------------------------------------------------------------
"""
    FixedEffects(y0,x0,FEType=:id)

Demeans y0 and x0 by taking out individual and/or time fixed effects.
Loops to handle unbalanced panels.


# Input
- `y0::Matrix`:      TxN matrix of dependent variables (possibly filled with NaNs), where T is the number of dates and N the number of cross-sectional units
- `x0::Array`:       TxKxN array with regressors, notice that regressor k for cross-sectional unit i is in x0[:,k,i]
- `FEType::Symbol`:  (:id,:idt, or :t) for individual fixed effects, :idt for individual and time fixed effects, :t for time fixed effects

# Output
- `y::Matrix`:
- `x::Array`:
- `yxAvg::Named tuple`:  yAvg_i,xAvg_i,yAvg_t,xAvg_t,yAvg,xAvg

# Notice
- for TxKxN -> TxNxK (or vice versa), do permutedims(z,[1,3,2])

  Paul.Soderlind@unisg.ch

"""
function FixedEffects(y0,x0,FEType=:id)

  (T,N) = (size(y0,1),size(y0,2))
  K     = size(x0,2)            #TxKxN

  vvt = falses(T,N)
  for i = 1:N
    vvt[:,i] = FindNNPs(view(y0,:,i),view(x0,:,:,i))    #use (t,i) data or not
  end

  if in(FEType,[:id,:idt])                            #individual fixed effects
    (yAvg_i,xAvg_i) = (fill(NaN,1,N),fill(NaN,1,K,N))
    for i = 1:N              #loop over cross section
      vv            = vvt[:,i]
      yAvg_i[1,i]   = mean(view(y0,vv,i))
      xAvg_i[1,:,i] = mean(view(x0,vv,:,i),dims=1)
    end
  end

  if in(FEType,[:idt,:t])                              #time fixed effects
    (yAvg_t,xAvg_t) = (fill(NaN,T),fill(NaN,T,K))
    for t = 1:T              #loop over time
      vv          = vvt[t,:]
      yAvg_t[t]   = mean(view(y0,t,vv))
      xAvg_t[t,:] = mean(view(x0,t,:,vv),dims=2)
    end
    yAvg = mean(y0[vvt])                                  #grand mean of y
    xAvg = permutedims([mean(x0[:,k,:][vvt]) for k=1:K])  #grand means of each of x
  end

  if FEType == :id
    y = y0 .- yAvg_i                     #subtract fixed individual effects
    x = x0 .- xAvg_i
    (yAvg_t,xAvg_t,yAvg,xAvg,N_t) = (NaN,NaN,NaN,NaN,NaN)
  elseif FEType == :idt
    y = y0 .- yAvg_i .- yAvg_t .+ yAvg   #subtract individual and time fixed effects
    x = x0 .- xAvg_i .- xAvg_t .+ xAvg
  elseif FEType == :t
    y = y0 .- yAvg_t                     #subtract time fixed effects
    x = x0 .- xAvg_t
    (yAvg_i,xAvg_i) = (NaN,NaN)
  end

  yxAvg = (;yAvg_i,xAvg_i,yAvg_t,xAvg_t,yAvg,xAvg,vvt)

  return y, x, yxAvg

end
#------------------------------------------------------------------------------
