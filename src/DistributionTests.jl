"""
    BeraJarqueTest(x)

Calculate the JB test for each column in a matrix. Reports `(skewness,kurtosis,JB)`.

"""
function BeraJarqueTest(x)
    (T,n) = (size(x,1),size(x,2))    #number of columns in x
    μ     = mean(x,dims=1)
    σ     = std(x,dims=1)
    xStd  = (x .- μ)./σ               #first normalize to a zero mean, unit std variable
    skewness = mean(z->z^3,xStd,dims=1)
    kurtosis = mean(z->z^4,xStd,dims=1)
    JB       = (T/6)*abs2.(skewness) + (T/24)*abs2.(kurtosis.-3)   #Jarque-Bera, Chisq(2)
    if n == 1 
        (skewness,kurtosis,JB) = (only(skewness),only(kurtosis),only(JB))  #to numbers if n=1
    end

    pskew = 2*ccdf.(Normal(0,sqrt(6/T)), abs.(skewness))
    pkurt = 2*ccdf.(Normal(3,sqrt(24/T)),abs.(kurtosis))
    pJB   = ccdf.(Chisq(2),JB)
    pvals = (;pskew,pkurt,pJB)

    return skewness, kurtosis, JB, pvals
end   


"""
    KolSmirTest(x1,TheoryCdf::Function)

Calculate the Kolmogorov-Smirnov test

### Output
- `KSstat::Float64`:     KS test statistic
- `xD::Number`:          x value with the largest diff beteen empirical and theoretical cdf

"""
function KolSmirTest(x1,TheoryCdf::Function)
    T            = length(x1)
    x1Sorted     = sort(x1)
    TheoryCdf_x  = TheoryCdf.(x1Sorted)
    edfH         = 1/T:1/T:1                        #empirical cdf for x1Sorted
    edfL         = 0:1/T:(1-1/T)
    D_candidates = abs.([edfH;edfL] - repeat(TheoryCdf_x,2))
    (D,vD)       = findmax(D_candidates)
    KSstat       = sqrt(T)*D
    xD           = repeat(x1Sorted,2)[vD]
    return KSstat, xD
end
