from speos.postprocessing.postprocessor import PostProcessor
from speos.utils.config import Config

config=Config()
config.parse_yaml("config_cardiovascular.yaml")

config.pp.tasks = config.pp.tasks[1:]
pp = PostProcessor(config)
pp.register_translation_table("./data/translation_table.tsv", hgnc_col="symbol", entrez_col="entrez_id", ensembl_col="ensembl_gene_id")
pp.load_outer_results("../results/cardiovascular5outer_results.json")

#pp.lof_intolerance(plot=False)
#pp.drugtarget()
#pp.mouseKO()
pp.go_enrichment()