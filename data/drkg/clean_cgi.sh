#!/bin/bash

grep "Gene:Compound" drkg.tsv > cgi.tsv
grep "Compound:Gene" drkg.tsv | awk -v FS="\t" -v OFS="\t" '{print $3, $2, $1}' >> cgi.tsv
grep -v "bioarx" cgi.tsv > tmp
mv tmp cgi.tsv