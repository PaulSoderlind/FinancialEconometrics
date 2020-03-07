#------------------------------------------------------------------------------
"""
    printmat([fh::IO],x;width=10,prec=3,NoPrinting=false,StringFmt="")

Print all elements of matrix with predefined formatting.

# Input
- `fh::IO`:            (optional) file handle. If not supplied, prints to screen
- `x::Array`:          string, date or array to print
- `width::Int`:        (keyword) scalar, width of printed cells
- `prec::Int`:         (keyword) scalar, precision of printed cells
- `NoPrinting::Bool`:  (keyword) bool, true: no printing, just return formatted string
- `StringFmt::String`: (keyword) string, "", "html", "csv"

# Output
- str         (if NoPrinting) string, (otherwise nothing)

# Examples. Try printing the following arrays:
- x = [11 12;21 22]
- x = Any[1 "ab"; Date(2018,10,7) 3.14]

# Uses
- fmtNumPs

# To do
- use Dict() for the options, width etc?


Paul.Soderlind@unisg.ch

"""
function printmat(fh::IO,x;width=10,prec=3,NoPrinting=false,StringFmt="")

  if isa(x,Union{String,Date,DateTime,Missing})  #eg. a single Date
    str = string(lpad(x,width),"\n")
    if NoPrinting
      return str
    else
      print(fh,str,"\n")
      return nothing
    end
  elseif isa(x,Nothing)
    return nothing
  end

  if ndims(x) > 2
    @warn("more than 2 dimensions")
    return nothing
  end

  (m,n) = (size(x,1),size(x,2))

  iob = IOBuffer()
  for i = 1:m                #loop over lines
    for j = 1:n-1                #loop over columns 1:n-1
      writeElementPs(iob,x,i,j,width,prec,StringFmt)
    end
    if StringFmt == "csv"                         #last (n) column
      writeElementPs(iob,x,i,n,width,prec,"")     #no , at end of line
    else
      writeElementPs(iob,x,i,n,width,prec,StringFmt)
    end
    write(iob,"\n")            #newline
  end
  str = String(take!(iob))

  if NoPrinting                              #no printing, just return str
    return str
  else                                       #print, return nothing
    print(fh,str,"\n")
    return nothing
  end

end
                  #when fh is not supplied: printing to screen
printmat(x;width=10,prec=3,NoPrinting=false,StringFmt="") = printmat(stdout::IO,
         x,width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printmat2

Call on printmat twice: to print to screen and then to an open file (IOStream)
"""
function printmat2(fh,x;width=10,prec=3,NoPrinting=false,StringFmt="")
  printmat(x,width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt)       #to screen
  if isa(fh,IOStream) && isopen(fh)
    printmat(fh,x,width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt)  #to file
  end
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    writeElementPs(iob,x,i,j,width,prec,StringFmt)

Writes one element to iob, formatting depends on type
"""
function writeElementPs(iob,x,i,j,width,prec,StringFmt)

  if isa(x[i,j],AbstractFloat)         #Float
    write(iob,fmtNumPs(x[i,j],width,prec,"right",StringFmt))
  elseif isa(x[i,j],Union{Int,String}) #Int, String
    write(iob,fmtNumPs(x[i,j],width,0,"right",StringFmt))
  elseif isa(x[i,j],Bool)              #Bool, BitArrays, as 0/1, left
    write(iob,fmtNumPs(x[i,j]+0,width,0,"right",StringFmt))
  elseif isa(x[i,j],Nothing)           #Nothing, as ""
    write(iob,fmtNumPs("",width,0,"right",StringFmt))
  else                                 #other types (Missing,Date,...), right
    write(iob,fmtNumPs(x[i,j],width,0,"right",StringFmt))
  end

  return nothing
 end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    fmtNumPs(z,width=10,prec=2,justify="right",StringFmt="")

Create a formatted string of a number. With prec=0, it can be used Bools and Strings



# Remark
- with prec=0, the function can be used for non-floats (incl. Bools and Strings)
- The Formatting.jl package provides more elegant solutions:
  fmt  = FormatSpec(string(">",width,".",prec,"f"))   #right justified, else "<"
  fmt  = FormatSpec(string(">",wid,"d"))              #for Int
  str  = Formatting.fmt(fmt1,z))

"""
function fmtNumPs(z,width=10,prec=2,justify="right",StringFmt="")

  if prec > 0                                     #if decimal number
    z   = round(z,digits=prec)                    #101.23
    str = split(string(z),'.')
    if length(str) > 1
      strR  = string(".",rpad(str[2],prec,"0"))   #.23
      strLR = string(str[1],strR)                 #"101" * ".23"
    else                                          #eg. NaN, missing
      strLR = string(z)
    end
  else
    isa(z,AbstractFloat) && (z = round(Int,z))    #for Floats
    strLR = string(z)                             #
  end

  if justify == "left"                            #justification
    strLR = rpad(strLR,width)
  else
    strLR = lpad(strLR,width)
  end

  if StringFmt == "html"                          #html or csv formatting
    strLR = string("<td>",strLR,"</td>")
  elseif StringFmt == "csv"
    strLR = string(strLR,",")
  end

  return strLR

end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printlnPs([fh::IO],z...;width=10,prec=3)

Subsitute for println, with predefined formatting.


# Input
- `fh::IO`:    (optional) file handle. If not supplied, prints to screen
- `z::String`: string, numbers and arrays to print

Paul.Soderlind@unisg.ch

"""
function printlnPs(fh::IO,z...;width=10,prec=3)

  for x in z                              #loop over inputs in z...
    if isa(x,Union{String,Date,DateTime,Missing})
      print(fh,lpad(x,width))
    elseif isa(x,Nothing)
      print(fh,"")
    else                                         #other types
      iob = IOBuffer()
      for i = 1:length(x)
        if isa(x[i],AbstractFloat)               #Float
          write(iob,fmtNumPs(x[i],width,prec,"right"))
        elseif isa(x[i],Nothing)                 #Nothing
          write(iob,lpad("",width))
        else                                     #Integer, etc
          write(iob,lpad(x[i],width))
        end
      end
      print(fh,String(take!(iob)))
    end
  end

  print(fh,"\n")

end
                      #when fh is not supplied: printing to screen
printlnPs(z...;width=10,prec=3) = printlnPs(stdout::IO,z...,width=width,prec=prec)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    println2Ps

Call on printlnPs twice: to print to screen and then to an open file (IOStream)
"""
function println2Ps(fh::IO,z...;width=10,prec=3)
  printlnPs(z...,width=width,prec=prec)              #to screen
  if isa(fh,IOStream) && isopen(fh)
    printlnPs(fh::IO,z...,width=width,prec=prec)     #to file
  end
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
function printblue(x...)
  foreach(z->printstyled(z,color=:blue,bold=true),x)
  print("\n")
end
function printred(x...)
  foreach(z->printstyled(z,color=:red,bold=true),x)
  print("\n")
end
function printmagenta(x...)
  foreach(z->printstyled(z,color=:magenta,bold=true),x)
  print("\n")
end
function printyellow(x...)
  foreach(z->printstyled(z,color=:yellow,bold=true),x)
  print("\n")
end
#------------------------------------------------------------------------------
