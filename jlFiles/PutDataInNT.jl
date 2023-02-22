"""
    PutDataInNT(x,header)

Creates a NamedTuple with, for instance, `X.a`, `X.b` and `X.c` where `x` is a matrix and `header = ["a" "b" "c"]`.

"""
function PutDataInNT(x,header)
    namesB = tuple(Symbol.(header)...)                            #a tuple (:a,:b,:b)
    X      = NamedTuple{namesB}([x[:,i] for i=1:size(x,2)])       #NamedTuple with X.a, X.b and X.c
    return X
end
