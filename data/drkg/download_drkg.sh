#!/bin/bash

mkdir temp
cd temp

wget  https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz 

tar xzf drkg.tar.gz
cd ..

grep "Gene:Compound" temp/drkg.tsv > cgi.tsv
grep "Compound:Gene" temp/drkg.tsv | awk -v FS="\t" -v OFS="\t" '{print $3, $2, $1}' >> cgi.tsv
grep -v "bioarx" cgi.tsv > tmp
mv tmp cgi.tsv
rm -rf temp