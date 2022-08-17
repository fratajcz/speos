import pandas as pd
import numpy as np

genes = pd.read_csv("gene_sets/mendelian_genes_hg38.bed", sep= " ")

chromosomes = genes["chromosome_name"].tolist()
symbols = genes["hgnc_symbol"].tolist()
starts = genes["start_position"].tolist()
ends = genes["end_position"].tolist()

neighbourhood_size = 60000

chromosomes_already_visited = []
chromosomes_in_collection = []

gene2snps = {gene: [] for gene in genes["hgnc_symbol"]}

#gwas_file = "/Users/florin.ratajczak/ppi-core-genes/data/imputed_Astle_et_al_2016_Red_blood_cell_count.txt"
gwas_file = "/Users/florin.ratajczak/ppi-core-genes/data/imputed_sorted"

with open(gwas_file, "r") as file:
    i = 0
    file.readline()
    for line in file:
        try:
            snp_position = int(line.split()[3]) 
        except:
            continue
        snp_chromosome = line.split()[2][3:]
        if snp_chromosome not in chromosomes_already_visited:
            chromosomes_already_visited.append(snp_chromosome)
            #print(snp_chromosome)
        while i < len(starts):
            gene_chromosome = chromosomes[i]
            if snp_chromosome < gene_chromosome: #if the gene is on the next chromosome
                break                            # get the next snp
            elif snp_chromosome > gene_chromosome: # if the snp is on the next chromosome
                i += 1                              # get the next gene
                continue
            if snp_position < starts[i] - neighbourhood_size:   # if the snp is before the gene 
                break                                           # get the next snp
            elif snp_position > ends[i] + neighbourhood_size:   # if the snp is after the gene
                i += 1                                          # gene the next gene
            else:
                if snp_chromosome not in chromosomes_in_collection:
                    chromosomes_in_collection.append(snp_chromosome)
                    #print(snp_chromosome)
                if snp_chromosome == gene_chromosome:
                    gene2snps[symbols[i]].append(line.split()[0])
                else: 
                    i += 1
                    continue
                break

print(np.sum([len(x) for x in gene2snps.values()]))

print(np.mean([len(x) for x in gene2snps.values()]))

print(np.sum([len(x) == 0 for x in gene2snps.values()]))

print(len(list(gene2snps.items())))
