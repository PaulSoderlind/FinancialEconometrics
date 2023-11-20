
"""
    PutDataInNT(x,header)

Creates a NamedTuple with, for instance, `N.X`, `N.Y` and `N.Z` where `x` is a matrix
and `header = ["X" "Y" "Z"]` or `["X","Y","Z"]`.

"""
function PutDataInNT(x,header)
    namesB = tuple(Symbol.(header)...)                            #a tuple (:X,:Y,:Z)
    N      = NamedTuple{namesB}([x[:,i] for i=1:size(x,2)])       #NamedTuple with N.X, N.Y and N.Z
    return N
end
