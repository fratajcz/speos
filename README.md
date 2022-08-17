# Identification of core genes from PPI and GWAS

The idea is to use PPI as a network scaffold and identify core genes as those
genes that are connected to many GWAS genes for each trait.

A first approach to do this would be to go through all candidate genes with
a certain minimal number of neighbors. For each candidate define discrete gene 
sets by neighborhood or quantitative by proximity in the network. Then use
either the discrete gene set or quantitive promximity as predictor in magma
gene set analysis.
