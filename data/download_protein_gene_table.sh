#!/bin/bash
#
# Download the table to translate from Entrez Protein Identifiers to Entrez Gene Identifiers.

wget https://ftp.ensembl.org/pub/release-109/tsv/homo_sapiens/Homo_sapiens.GRCh38.109.entrez.tsv.gz 

gunzip Homo_sapiens.GRCh38.109.entrez.tsv.gz
head -n 1 Homo_sapiens.GRCh38.109.entrez.tsv > tmp
grep "ENSP" Homo_sapiens.GRCh38.109.entrez.tsv | sort >> tmp
awk -v OFS="\t" '{print $3, $1}' tmp > protein_gene_table.tsv

rm tmp
rm Homo_sapiens.GRCh38.109.entrez.tsv