from speos.postprocessing.ldblocks import LDBlockChecker
from extensions.preprocessing import preprocess_labels
from scipy.stats import fisher_exact


checker = LDBlockChecker(snpfile="/mnt/storage/prs/ldblocks/BW/BW_sign_snps.bed")

checker.build_ldblocks()
checker.build_genes()
checker.build_snps()
checker.build_coregenes("/mnt/storage/speos/results/bw_film_nohetioouter_results.json", cs=1)

mendelians = preprocess_labels("/home/ubuntu/speos/extensions/bw_only_genes.tsv")

checker.build_mendelians(mendelians)


checker.assign_genes_to_ld_block()
checker.assign_snps_to_ld_block(offset=1e4)
checker.assign_coregenes_to_ld_block()
checker.assign_mendelians_to_ld_block()

#print(checker.check_overlap())


array, table = checker.count_ldblocks(cs = 11)
table.to_csv("BW_only_ldblocks_by_snps.tsv", sep="\t")
print(array)
print(fisher_exact(array))
