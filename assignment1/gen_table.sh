#!/bin/bash
while IFS= read -r line;do
    echo -e '\hline' >> table.tex
    linearray=($line)
    echo "${linearray[1]} & ${linearray[0]} \\\\" >> table.tex
done < "test.txt"

#awk -F"\t" '{print "\\hline\n" $2 " & "  $1 "\\\\"}' test.txt