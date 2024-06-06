#!/bin/bash

wget https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.chr.gtf.gz 

gunzip Homo_sapiens.GRCh38.110.chr.gtf.gz

grep "protein_coding" Homo_sapiens.GRCh38.110.chr.gtf > tmp

awk -v FS="\t" -v OFS="\t" '{if ($3 == "gene") {split ($9, a, " "); split(a[6], b, "\""); print "chr" $1, $4, $5, b[2]}}' tmp > GRCH38.110.genes

grep -v chrMT GRCH38.110.genes > tmp

grep -v chrX tmp > GRCH38.110.genes

grep -v chrY GRCH38.110.genes > tmp