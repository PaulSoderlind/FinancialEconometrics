"""
    DrawBlocks(T,BlockSize)

Draw a T-vector of indices `v` that can be used to create bootstrap residuals. 
The indices are such that they form blocks of length `BlockSize`.

"""
function DrawBlocks(T,BlockSize)
    nBlocks = cld(T,BlockSize)                 #number of blocks, rounded up
    v0      = rand(1:T,nBlocks)                #nBlocks, random starting obs of blocks
    v       = vec(v0' .+ vec(0:BlockSize-1))   #each block in a column
    v       = replace(z -> z>T ? z-T : z,v)    #wrap around if index > T
    #println(v)                                #uncomment to see result
    return v
end
