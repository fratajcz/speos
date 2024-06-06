from speos.postprocessing.ldblocks import LDBlockChecker
from extensions.preprocessing import preprocess_labels
from scipy.stats import fisher_exact

checker = LDBlockChecker(snpfile="/mnt/storage/prs/ldblocks/Liu_UC_snps_5e-8.bed")

checker.build_ldblocks()
checker.build_genes()
checker.build_snps()
checker.build_coregenes("/mnt/storage/speos/results/uc_film_nohetioouter_results.json", cs=1)

mendelians = preprocess_labels("/home/ubuntu/speos/extensions/uc_only_genes.tsv")

checker.build_mendelians(mendelians)


checker.assign_genes_to_ld_block()
checker.assign_snps_to_ld_block()
checker.assign_coregenes_to_ld_block()
checker.assign_mendelians_to_ld_block()

#print(checker.check_overlap())

array, table = checker.count_ldblocks(cs = 11)
table.to_csv("UC_only_ldblocks_by_snps.tsv", sep="\t")
print(fisher_exact(array))
#print(checker.count_ldblocks(normalize=True, cs = 11))
#print(fisher_exact(checker.count_ldblocks(normalize=True, cs = 11)[0]))

checker = LDBlockChecker(snpfile="notebooks/UC_film_nohetio_master_regulators.csv", snp_is_bed=False)

checker.build_ldblocks()
checker.build_genes()
checker.build_snps()
checker.build_coregenes("/mnt/storage/speos/results/uc_film_nohetioouter_results.json", cs=1)

checker.build_mendelians(mendelians)

checker.assign_genes_to_ld_block()
checker.assign_snps_to_ld_block()
checker.assign_coregenes_to_ld_block()
checker.assign_mendelians_to_ld_block()

array, table = checker.count_ldblocks(cs=11)
table.to_csv("UC_only_ldblocks_by_master_regulators.tsv", sep="\t")
print(fisher_exact(array))
print(checker.count_ldblocks(normalize=True, cs = 11))
print(fisher_exact(checker.count_ldblocks(normalize=True, cs = 11)[0]))