"""
    printTable([fh::IO],x,colNames=[],rowNames=[];
               width=10,prec=3,NoPrinting=false,StringFmt="",cell00="")

Print formatted table with row names (1st column) column names (1st row), and data matrix (the rest).


# Input
- `fh::IO`:            (optional) file handle. If not supplied, prints to screen
- `x::Array`:          string, date or array to print
- `width::Int`:        (keyword) scalar, width of printed cells. [10]
- `prec::Int`:         (keyword) scalar, precision of printed cells. [3]
- `NoPrinting::Bool`:  (keyword) bool, true: no printing, just return formatted string [false]
- `StringFmt::String`: (keyword) string, "", "html", "csv"
- `cell00::String`:    (keyword) string, for row 0, column 0

# Output
- `str::String`:      (if NoPrinting) string, (otherwise nothing)

# Example
```
xA = [1 "ab" "abc"; "ccc" 3.14 missing]
printTable(xA,colNames,["1";"4"],width=12,prec=2)
```

# Uses
- printmat

"""
function printTable(fh::IO,x,colNames=[],rowNames=[];
                    width=10,prec=3,NoPrinting=false,StringFmt="",cell00="")

  isempty(x) && return nothing                        #do nothing is isempty(x)

  (m,n) = (size(x,1),size(x,2))

  if isempty(rowNames)                                 #create row names "r1"
    rowNames = [string("r",i) for i = 1:m]
  end
  if isempty(colNames)                                 #create column names "c1"
    colNames = [string("c",i) for i = 1:n]
  end

  rNamesWidth = maximum([length(rowNames[i]) for i = 1:length(rowNames)])  #max length of rowNames
  rNamesWidth = max(rNamesWidth,length(cell00))

  iob = IOBuffer()
  if StringFmt == "html"                                             #print column names
    write(iob,"<tr><th>",lpad(cell00,rNamesWidth),"</th>")
    for i = 1:n
      write(iob,"<th>",lpad(colNames[i],width),"</th>")
    end
    write(iob,"</tr>")
  elseif StringFmt == "csv"
    write(iob,lpad(string(cell00,","),rNamesWidth))
    for i = 1:n-1
      write(iob,lpad(colNames[i],width),",")
    end
    write(iob,lpad(colNames[n],width))       #no , at line end
  else
    write(iob,lpad(cell00,rNamesWidth))                  #cell 0,0
    for i = 1:n                                          #create string
      write(iob,lpad(colNames[i],width))
    end
  end
  write(iob,"\n")

  xStr  = printmat(fh,x,width=width,prec=prec,NoPrinting=true,StringFmt=StringFmt)   #body of table, one long string
  xStrV = split(xStr,"\n")                       #vector of strings (one per row of x)

  for i = 1:m                           #loop over rows in x, print rowNames[i] and x[i,:]
    if StringFmt == "html"
      write(iob,"<tr><td><b>",rpad(rowNames[i],rNamesWidth),"</td></b>",xStrV[i],"</tr> \n")
    elseif StringFmt == "csv"
      write(iob,rpad(string(rowNames[i],","),rNamesWidth),xStrV[i],"\n")
    else
      write(iob,rpad(rowNames[i],rNamesWidth),xStrV[i],"\n")
    end
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
printTable(x,colNames=[],rowNames=[];width=10,prec=3,NoPrinting=false,StringFmt="",cell00="") =
printTable(stdout::IO,x,colNames,rowNames,
                      width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt,cell00=cell00)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printTable2

Call on printTable2 twice: to print to screen and then to an open file (IOStream)
"""
function printTable2(fh,x,colNames=[],rowNames=[];width=10,prec=3,NoPrinting=false,StringFmt="",cell00="")
  printTable(x,colNames,rowNames,width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt,cell00=cell00)      #to screen
  if isa(fh,IOStream) && isopen(fh)
    printTable(fh,x,colNames,rowNames,width=width,prec=prec,NoPrinting=NoPrinting,StringFmt=StringFmt,cell00=cell00) #to file
  end
end
#------------------------------------------------------------------------------
