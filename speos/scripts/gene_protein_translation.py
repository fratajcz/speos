from biomart import BiomartServer
import pandas as pd
"""
atts = ['ensembl_peptide_id', 'ensembl_gene_id']

server = BiomartServer("http://www.ensembl.org/biomart/martservice")
hge = server.datasets['hsapiens_gene_ensembl']

s = hge.search({'attributes': atts}, header=1)
good_lines = []
for i, l in enumerate(s.iter_lines()):
    l = l.decode('utf-8')
    if i == 0:
        good_lines.append(l + "\n")
    if "ENSP" in l and "ENSG" in l:
        good_lines.append(l + "\n")

with open("data/protein_gene_table.tsv", "w") as file:
    file.writelines(good_lines)
"""

table = pd.read_csv("data/protein_gene_table.tsv", header=0, sep="\t")

protein_to_gene_dict = {row[0]: row[1] for _, row in table.iterrows()}

print(protein_to_gene_dict)