##------------------------------------------------------------------------------
"""
    Gmm2MomFn(par,x)

Calculate traditional 2 moment conditions for estimating [μ,σ²]. Returns a Tx2 matrix

### Input
- `par::Vector`: [μ,σ²]
- `x::Vector`:   T-vector with data

### Output
- `g::Matrix`:    Tx2, moment conditions

"""
function Gmm2MomFn(par,x)
    (μ,σ²) = (par[1],par[2])
    g      = hcat(x .- μ, abs2.(x .- μ) .- σ²)  #Tx2
    return g
end
##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
"""
    Gmm4MomFn(par,x)

Calculate 4 moment conditions for estimating [μ,σ²]

### Input
- `par::Vector`: [μ,σ²]
- `x::Vector`:   T-vector with data

### Output
- `g::Matrix`:    Tx4, moment conditions

"""
function Gmm4MomFn(par,x)
  (μ,σ²) = (par[1],par[2])
  g      = hcat(x .- μ, (x .- μ).^2 .- σ², (x .- μ).^3, (x .- μ).^4 .- 3*σ²^2)    #Tx4
  return g
end
##------------------------------------------------------------------------------
