from speos.postprocessing.postprocessor import PostProcessor
from speos.utils.config import Config


def main():
    config = Config()
    config.parse_yaml("config_immune_dysregulation.yaml")
    postprocessor = PostProcessor(config)

    pval_tp_fp, pval_tn_fp, plot = postprocessor.compare(compare_file = "data/forweb_cleaned_exac_r03_march16_z_data_pLI.txt",
                                       compare_column = "pLI",
                                       join_column_compare = "gene",
                                       join_column_results = "hgnc")

    plot.savefig("pLI_cardiovascular.png")

if __name__ == "__main__":
    main()