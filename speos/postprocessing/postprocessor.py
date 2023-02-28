import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from collections import Counter
import numpy as np
import logging
import os
from speos.utils.logger import setup_logger
import random
import json
from speos.postprocessing.pp_utils import PostProcessingTable
import speos.utils.path_utils as pu
import traceback

class PostProcessor:
    """Reads a results file and generates reports and analyses on it. The results file must contain identifers, labels and predictions per gene"""

    def __init__(self, config, translation_table="./data/hgnc_official_list.tsv"):
        self.config = config
        self.logger_args = [config, __name__]
        self.num_runs_for_random_experiments = 1000
        self.outer_result = None   # is populated by overlap_analysis when pp is run

        self.register_translation_table(translation_table)

        self.target = self.config.input.tag
        self.consensus = self.config.pp.consensus        

    def init_pp_table(self):
        unknown_genes, all_genes, positive_genes = self.get_unknown_genes()
        self.pp_table = PostProcessingTable(index=all_genes)
        included_genes = self._read_results_file()["hgnc"].tolist()
        self.pp_table.add("Is Included", index=included_genes, values=True, remaining=False)
        self.pp_table.add("Mendelian", index=positive_genes, values=True, remaining=False)

    def save_pp_table(self):
        path = os.path.join(pu.postprocessing_results_path(self.config), self.config.name + "_pp_table.tsv")
        self.pp_table.save(path)

    def run(self):
        """Runs all tasks that are specified in the config as pp.tasks
        
            Returns:
                list: The results of all individual tasks, in the same order as specified in the config file.
                
        """

        save = self.config.pp.save

        self.init_pp_table()
        logger = setup_logger(*self.logger_args)

        # ensure overlap analysis is run first
        if "overlap_analysis" in self.config.pp.tasks:
            logger.info("Applying concensus strategy: {}".format(self.consensus))
            value = [self.overlap_analysis()]
            self.config.pp.tasks.remove("overlap_analysis")

        if self.config.pp.tasks is not None:
            for function in self.config.pp.tasks:
                try:
                    value.append(getattr(self, function)())
                except FileNotFoundError as e:
                    logger.error("FileNotFoundError during handling of postprocessing task {}:".format(function))
                    logger.error(e)
                    continue
                except KeyError as e:
                    logger.error("KeyError during handling of postprocessing task {}:".format(function))
                    logger.error(e)
                    continue
        
        if save:
            self.save_pp_table()

        return value

    def register_translation_table(self, path_to_translation_table, hgnc_col="symbol", entrez_col="entrez_id", ensembl_col="ensembl_gene_id"):
        self.path_to_translation_table = path_to_translation_table
        self.hgnc_col = hgnc_col
        self.entrez_col = entrez_col
        self.ensembl_col = ensembl_col

    def pathway(self, results_path=None, plot=True, save=True) -> pd.DataFrame:
        """ 
            Runs Pathway enrichment on the results of the outer crossvalidation.
            Uses only unknown genes as background, mendelians are removed.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                pandas.DataFrame: contains the results of the Pathway Enrichment Analysis

        """

        from speos.postprocessing.goea import GOEA_Study
        logger = setup_logger(*self.logger_args)
        logger.info("Starting Pathway Enrichment Analysis.")

        if self.outer_result is None:
            logger.warning("Pathway Enrichment Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)

        translation_table = self.get_translation_table()
        entrez2symbol = {line[1]: line[0] for i, line in translation_table.iterrows()}

        path = "./data/pathways/wikipathways-20220710-gmt-Homo_sapiens.gmt"

        pathway2symbol = {}
        pathway2name = {}

        with open(path, "r") as file:
            for line in file.readlines():
                fields = line.split("\t")
                pathway = fields[0].split("%")[2]
                name = fields[0].split("%")[0]
                entrez_ids = [int(id.strip()) for id in fields[2:]]
                pathway2symbol.update({pathway: [entrez2symbol[entrez_id] for entrez_id in entrez_ids if entrez_id in entrez2symbol.keys()]})
                pathway2name.update({pathway: name})

        goea = GOEA_Study()

        goea.set_term_description(pathway2name)
        goea.set_term_symbol(pathway2symbol)

        task = ""

        df = goea.analyze(list(self.outer_result[0].keys()), set(unknown_genes), task)

        if self.config.pp.save:
            self.create_if_not_exists(self.config.pp.save_dir)
            tsv_path = os.path.join(self.config.pp.save_dir, self.config.name + "_pathwayea.tsv")
            logger.info("Found {} significant terms, writing table to {}".format(len(df.index), tsv_path))
            df.to_csv(tsv_path, sep="\t")

        if self.config.pp.plot:
            self.create_if_not_exists(self.config.pp.plot_dir)
            image_path = os.path.join(self.config.pp.plot_dir, self.config.name + "_pathwayea.png")
            logger.info("Saving plot to {}".format(image_path))
            if len(df) > 0:
                try:
                    goea.plot(df, image_path)
                except Exception as e:
                    logger.error("Something went wrong trying to plot:")
                    logger.error(traceback.format_exc())
            else:
                logger.info("Not Plotting Pathway Enrichment because no significant Terms have been found.")

        return df

    def dge(self, results_path=None, plot=True, save=True, convergence_score=1) -> pd.DataFrame:
        """ Runs Differential Gene Expression enrichment on the results of the outer crossvalidation.
            Uses only unknown genes as background, mendelians are removed.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                pandas.DataFrame: contains the results of the Differential Gene Expression Enrichment Analysis.
        """

        import yaml
        from scipy.stats import fisher_exact

        logger = setup_logger(*self.logger_args)
        logger.info("Starting Differential Gene Expression Enrichment Analysis.")

        if self.outer_result is None:
            logger.warning("Differential Gene Expression Enrichment Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        phenotype = self.config.input.tag
        with open("./data/dge/mapping.yaml", "r") as file:
            mapping = yaml.load(file, Loader=yaml.SafeLoader)

        for option in mapping.keys():
            if option.startswith(phenotype.lower()):
                phenotype = option
        
        try:
            subtypes = mapping[phenotype.lower()]
        except KeyError:
            logger.warning("Phenotype {} not registered for differential gene expression analysis.".format(phenotype))
            return

        num_phenotypes = len(list(subtypes.keys()))

        logger.info("Found {} subtypes for phenotype {}: {}.".format(num_phenotypes, phenotype, list(subtypes.keys())))

        if num_phenotypes == 0:
            logger.info("Skipping differentially expressed genes analysis.")
            return None
        elif num_phenotypes == 1:
            phenotypes = list(subtypes.keys())
            value_list = list(subtypes.values())
        else:
            phenotypes = list(subtypes.keys()) + ["Union"]
            value_list = list(subtypes.values()) + [""]


        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)
        results = pd.DataFrame(index=phenotypes)

        mendelian_odds_ratios = []
        mendelian_pvals = []
        candidate_odds_ratios = []
        candidate_pvals = []
        n_dge = []
        n_mendelians_with_de = []
        n_mendelians_without_de = []
        n_nonmendelians_with_de = []
        n_nonmendelians_without_de = []
        n_candidate_with_de = []
        n_candidate_without_de = []
        n_noncandidate_with_de = []
        n_noncandidate_without_de = []

        valid_union_genes = set()
        unknown_union_genes = set()
        dge_genes_union = set()

        for subtype, values in zip(phenotypes, value_list):
            
            if subtype != "Union":
                dge_genes = set(pd.read_csv(values["file"], header=0, comment='#', index_col=False, sep="\t")["Symbol"].to_list())
                dge_genes_union.update(dge_genes)
                valid_dge_genes = self._return_only_valid(dge_genes, all_genes)
                valid_union_genes.update(valid_dge_genes)
                unknown_dge_genes = self._return_only_valid(dge_genes, unknown_genes)
                unknown_union_genes.update(unknown_dge_genes)
            else:
                dge_genes = dge_genes_union
                unknown_dge_genes = unknown_union_genes
                valid_dge_genes = valid_union_genes

            self.pp_table.add("DGE: {}".format(subtype), valid_dge_genes, True, False)
            array = self.make_contingency_table(all_genes, positive_genes, valid_dge_genes)
            n_mendelians_with_de.append(array[0][0].item())
            n_mendelians_without_de.append(array[1][0].item())
            n_nonmendelians_with_de.append(array[0][1].item())
            n_nonmendelians_without_de.append(array[1][1].item())
            is_enriched_result = fisher_exact(array)

            logger.info("Total of {} {} DE genes, {} of them match with our translation table.".format(len(dge_genes), subtype, len(valid_dge_genes)))
            logger.info("Found {} {} DE genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
                    len(valid_dge_genes.intersection(positive_genes)), subtype, len(positive_genes), is_enriched_result[1], round(is_enriched_result[0], 3), len(unknown_dge_genes), len(unknown_genes)))

            mendelian_odds_ratios.append(is_enriched_result[0])
            mendelian_pvals.append(is_enriched_result[1])
            n_dge.append(len(valid_dge_genes))
            
            

            predicted_genes = set([key for key, value in self.outer_result[0].items() if value >= convergence_score])
            array = self.make_contingency_table(unknown_genes, predicted_genes, unknown_dge_genes)
            n_candidate_with_de.append(array[0][0].item())
            n_candidate_without_de.append(array[1][0].item())
            n_noncandidate_with_de.append(array[0][1].item())
            n_noncandidate_without_de.append(array[1][1].item())
            is_enriched_result = fisher_exact(array)

            logger.info("Fishers Exact Test for {} DE genes among Predicted Genes. p: {:.2e}, OR: {}".format(subtype, is_enriched_result[1], round(is_enriched_result[0], 3)))
            logger.info("{} DE genes Confusion Matrix:\n".format(subtype) + str(array))

            candidate_odds_ratios.append(is_enriched_result[0])
            candidate_pvals.append(is_enriched_result[1])

        results["Mendelian ORs"] = mendelian_odds_ratios
        results["Mendelian pvals"] = mendelian_pvals
        results["Candidate ORs"] = candidate_odds_ratios
        results["Candidate pvals"] = candidate_pvals
        results["N DEG"] = n_dge
        results["N Mendelian And DEG"] = n_mendelians_with_de
        results["N Mendelian Not DEG"] = n_mendelians_without_de
        results["N Not Mendelian And DEG"] = n_nonmendelians_with_de
        results["N Not Mendelian Not DEG"] = n_nonmendelians_without_de
        results["N Candidate And DEG"] = n_candidate_with_de
        results["N Candidate Not DEG"] = n_candidate_without_de
        results["N Not Candidate And DEG"] = n_noncandidate_with_de
        results["N Not Candidate Not DEG"] = n_noncandidate_without_de
        return results

    def hpo_enrichment(self, results_path=None, plot=True, save=True) -> pd.DataFrame:
        """ Runs HPO Term enrichment on the results of the outer crossvalidation.
            Uses only unknown genes as background, mendelians are removed.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                pandas.DataFrame: contains the results of the HPO Enrichment Analysis
        """

        from speos.postprocessing.goea import GOEA_Study

        logger = setup_logger(*self.logger_args)
        logger.info("Starting HPO Enrichment Analysis.")

        if self.outer_result is None:
            logger.warning("HPO Enrichment Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)

        translation_table = self.get_translation_table()
        entrez2symbol = {line[1]: line[0] for i, line in translation_table.iterrows()}

        path = "./data/hpo/genes_to_phenotype.txt"

        df = pd.read_csv(path, skiprows=1, sep="\t", usecols=[0, 2, 3], names=["entrez", "hpo", "description"])

        hpo2name = {line[1]: line[2].split(" - ")[0].strip() for i, line in df.iterrows()}
        hpo2symbol = {}
        for i, line in df.iterrows():
            if int(line[0]) in entrez2symbol.keys():
                if line[1] in hpo2symbol.keys():
                    hpo2symbol[line[1]].append(entrez2symbol[int(line[0])])
                else:
                    hpo2symbol[line[1]] = [entrez2symbol[int(line[0])]]

        goea = GOEA_Study()

        goea.set_term_description(hpo2name)
        goea.set_term_symbol(hpo2symbol)

        task = ""

        df = goea.analyze(list(self.outer_result[0].keys()), set(unknown_genes), task)

        if save:
            self.create_if_not_exists(self.config.pp.save_dir)
            tsv_path = os.path.join(self.config.pp.save_dir, self.config.name + "_hpoea.tsv")
            logger.info("Found {} significant terms, writing table to {}".format(len(df.index), tsv_path))
            df.to_csv(tsv_path, sep="\t")

        if plot:
            self.create_if_not_exists(self.config.pp.plot_dir)
            image_path = os.path.join(self.config.pp.plot_dir, self.config.name + "_hpoea.png")
            if len(df) > 0:
                logger.info("Saving plot to {}".format(image_path))
                try:
                    goea.plot(df, image_path)
                except Exception as e:
                    logger.error("Something went wrong trying to plot:")
                    logger.error(traceback.format_exc())
            else:
                logger.info("Not Plotting HPO Enrichment because no significant Terms have been found.")

        return df

    def go_enrichment(self, results_path=None, plot=True, save=True) -> pd.DataFrame:
        """ Runs GO Term enrichment on the results of the outer crossvalidation.
            Uses only unknown genes as background, mendelians are removed.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                pandas.DataFrame: contains the results of the GO Term Enrichment Analysis
        """

        from speos.postprocessing.goea import GOEA_Study

        logger = setup_logger(*self.logger_args)
        logger.info("Starting GO Enrichment Analysis.")

        if self.outer_result is None:
            logger.warning("GO Enrichment Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)

        goea = GOEA_Study()

        for task in ["biological process", "molecular function", "cellular component"]:

            df = goea.analyze(list(self.outer_result[0].keys()), set(unknown_genes), task)

            if save:
                self.create_if_not_exists(self.config.pp.save_dir)
                path = os.path.join(self.config.pp.save_dir, self.config.name + "_goea_{}.tsv".format("_".join(task.split(" "))))
                logger.info("Found {} significant terms for task {}, writing table to {}".format(len(df.index), task, path))
                df.to_csv(path, sep="\t")

            if plot:
                self.create_if_not_exists(self.config.pp.plot_dir)
                path = os.path.join(self.config.pp.plot_dir, self.config.name + "_goea_{}.png".format("_".join(task.split(" "))))
                logger.info("Saving plot to {}".format(path))
                if len(df) > 0:
                    try:
                        goea.plot(df, path)
                    except Exception as e:
                        logger.error("Something went wrong trying to plot:")
                        logger.error(traceback.format_exc())
                else:
                    logger.info("Not Plotting Go Enrichment fo {} because no significant Terms have been found.".format(task))

            goea.reset()

        return df

    def drugtarget(self, results_path=None, plot=True, save=True) -> tuple:
        """ Takes the results of the outer crossvalidation and analyzes if there is an enrichment of drug targets among the predicted genes.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                tuple([...], pd.DataFrame): Returns a tuple of various results, most of which are summarized in the DataFrame at the end (tuple[-1]).

            """

        logger = setup_logger(*self.logger_args)

        if self.outer_result is None:
            logger.warning("Drug Target Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        from scipy.stats import fisher_exact, mannwhitneyu
        from speos.scripts.utils import fdr

        df = pd.DataFrame(columns=["Group Name", "Group N", "N Drug Targets", "OR DT", "pval DT unadjusted", "pval DT adjusted (FDR)", "Median # of DT", "xDC"," ", "Pairwse Comparison", "pval xDC unadjusted", "pval xDC adjusted (FDR)", "U-Stat"],
                          index=range(3))
        df["Group Name"] = ["Mendelian", "Candidate Gene", "Noncandidate Gene"]
        df[" "] = [" "] * 3

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)

        hgnc2degree = self.get_drugtarget_dict()
        if "Total" in self.outer_result[0].keys():
            self.outer_result[0].pop("Total")

        min_nr_of_predictions = 1

        hgnc2predictions = {hgnc: num_predictions for hgnc, num_predictions in self.outer_result[0].items() if num_predictions >= min_nr_of_predictions}

        predicted_genes = set(hgnc2predictions.keys())
        drug_targets = set(hgnc2degree.keys())
        not_predicted_genes = unknown_genes - predicted_genes

        df["Group N"] = [len(positive_genes), len(predicted_genes), len(not_predicted_genes)]

        unknown_drug_targets = self._return_only_valid(drug_targets, unknown_genes)
        valid_drug_targets = self._return_only_valid(drug_targets, all_genes)
        noncandidate_drug_targets = self._return_only_valid(drug_targets, not_predicted_genes)
        candidate_drug_targets = self._return_only_valid(drug_targets, predicted_genes)
        mendelian_drug_targets = self._return_only_valid(drug_targets, positive_genes)

        assert len(valid_drug_targets) == len(noncandidate_drug_targets) + len(candidate_drug_targets) + len(mendelian_drug_targets)

        df["N Drug Targets"] = [len(mendelian_drug_targets), len(candidate_drug_targets), len(noncandidate_drug_targets)]

        self.pp_table.add("Drug Target", valid_drug_targets, True, False)
        valid_dict = {gene: degree for gene, degree in hgnc2degree.items() if gene in all_genes}
        genes, degree = list(zip(*valid_dict.items()))
        self.pp_table.add("Number of Drug Interactions", genes, degree, 0)
        
        array = self.make_contingency_table(all_genes, positive_genes, valid_drug_targets)

        drug_target_results = []
        is_drug_target_result = fisher_exact(array)
        drug_target_results.append(is_drug_target_result)

        logger.info("Total of {} drug targets, {} of them match with our translation table.".format(len(drug_targets), len(drug_targets.intersection(all_genes))))
        logger.info("Found {} drug targets genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
            len(drug_targets.intersection(positive_genes)), len(positive_genes), is_drug_target_result[1], round(is_drug_target_result[0], 3), len(unknown_drug_targets), len(unknown_genes)))

        array = self.make_contingency_table(unknown_genes, predicted_genes, unknown_drug_targets)

        is_drug_target_result = fisher_exact(array)
        drug_target_results.append(is_drug_target_result)

        df["OR DT"] = [drug_target_results[0][0], drug_target_results[1][0], np.nan]
        df["pval DT unadjusted"] = [drug_target_results[0][1], drug_target_results[1][1], np.nan]

        logger.info("Fishers Exact Test for Drug Targets among Predicted Genes. p: {:.2e}, OR: {}".format(is_drug_target_result[1], round(is_drug_target_result[0], 3)))
        logger.info("Drug Targets Confusion Matrix:\n" + str(array))

        positive_genes_and_drug_targets = positive_genes.intersection(drug_targets)
        predicted_genes_and_drug_targets = predicted_genes.intersection(drug_targets)
        not_predicted_genes_and_drug_targets = not_predicted_genes.intersection(drug_targets)

        positive_degrees = [hgnc2degree[hgnc] for hgnc in positive_genes_and_drug_targets]
        predicted_degrees = [hgnc2degree[hgnc] for hgnc in predicted_genes_and_drug_targets]
        not_predicted_degrees = [hgnc2degree[hgnc] for hgnc in not_predicted_genes_and_drug_targets]

        df["Median # of DT"] = [np.median(positive_degrees), np.median(predicted_degrees), np.median(not_predicted_degrees)]
        df["xDC"] = [np.median(positive_degrees) / np.median(not_predicted_degrees), np.median(predicted_degrees) / np.median(not_predicted_degrees), 1]
        df["Pairwse Comparison"] = ["Mendelian vs Candidate Gene", "Mendelian vs Noncandidate Gene", "Candidate Gene vs Noncandidate Gene"]

        pvals = []
        u_stats = []
        drug_degree_result = mannwhitneyu(predicted_degrees,
                                          not_predicted_degrees)
        pvals.append(drug_degree_result[1])
        u_stats.append(drug_degree_result[0])

        drug_degree_result = mannwhitneyu(positive_degrees,
                                          not_predicted_degrees)
        pvals.append(drug_degree_result[1])
        u_stats.append(drug_degree_result[0])

        drug_degree_result = mannwhitneyu(positive_degrees,
                                          predicted_degrees)
        pvals.append(drug_degree_result[1])
        u_stats.append(drug_degree_result[0])

        df["pval xDC unadjusted"] = [pvals[2], pvals[1], pvals[0]]
        df["U-Stat"] = [u_stats[2], u_stats[1], u_stats[0]]

        pvals = fdr(pvals)
        df["pval xDC adjusted (FDR)"] = [pvals[2], pvals[1], pvals[0]]

        logger.info("U-Test for number of Drug interactions in Predicted Genes vs Non-Predicted Genes. q: {:.2e}, U: {}".format(pvals[0], round(u_stats[0], 3)))
        logger.info("U-Test for number of Drug interactions in Mendelian Genes vs Non-Predicted Genes. q: {:.2e}, U: {}".format(pvals[1], round(u_stats[1], 3)))
        logger.info("U-Test for number of Drug interactions in Mendelian Genes vs Predicted Genes. q: {:.2e}, U: {}".format(pvals[2], round(u_stats[2], 3)))

        logger.info("0, 25, 50, 75 and 99% quantiles for Mendelians: {}".format(np.quantile(positive_degrees, (0, 0.25, 0.5, 0.75, 0.99))))
        logger.info("0, 25, 50, 75 and 99% quantiles for Predicted Genes: {}".format(np.quantile(predicted_degrees, (0, 0.25, 0.5, 0.75, 0.99))))
        logger.info("0, 25, 50, 75 and 99% quantiles for Non-Predicted Genes: {}".format(np.quantile(not_predicted_degrees, (0, 0.25, 0.5, 0.75, 0.99))))

        if plot:
            self.make_boxplot(not_predicted_degrees, predicted_degrees, positive_degrees, plot=plot)

        if save:
                self.create_if_not_exists(self.config.pp.save_dir)
                path = os.path.join(self.config.pp.save_dir, self.config.name + "_drugtarget.tsv")
                df.to_csv(path, sep="\t")

        return drug_target_results, pvals, (not_predicted_degrees, predicted_degrees, positive_degrees), df

    def druggable(self, results_path=None, plot=False, save=True):
        """ Takes the results of the outer crossvalidation and analyzes if there is an enrichment of druggable genes among the predicted genes.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                tuple(list[ResultA, ResultB], list[ResultC, ResultD], pd.DataFrame): Returns a tuple of various results, most of which are summarized in the DataFrame at the end (tuple[-1]).
                    ResultA is the enrichment of druggable genes in positively labeled genes, ResultB is the enrichment in the candidates.
                    ResultC is the entrichment of druggable genes among the non-drug-target genes in the positively labeled genes, ResultsD is the same enrichment in the candidates.
        """


        logger = setup_logger(*self.logger_args)
        if self.outer_result is None:
            logger.warning("Druggable Gene Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        from scipy.stats import fisher_exact

        df = pd.DataFrame(columns=["Group Name", "Group N", "N Druggable", "OR Dr", "pval Dr unadjusted", "pval Dr adjusted (FDR)", "N Non-Drugtarget", "N Druggable among Non-Drugargets", "OR Dr-", "pval Dr- unadjusted", "pval Dr- adjusted (FDR)"],
                          index=range(3))
        df["Group Name"] = ["Mendelian", "Candidate Gene", "Noncandidate Gene"]

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)
        predicted_genes = set(self.outer_result[0].keys())
        not_predicted_genes = unknown_genes - predicted_genes
        df["Group N"] = [len(positive_genes), len(predicted_genes), len(not_predicted_genes)]

        druggable_genes = set(self.get_druggable_genes("./data/dgidb/druggable_genome.tsv"))
        unknown_druggable_genes = self._return_only_valid(druggable_genes, unknown_genes)
        positive_druggable_genes = self._return_only_valid(druggable_genes, positive_genes)
        predicted_druggable_genes = self._return_only_valid(druggable_genes, predicted_genes)
        noncandidate_druggable_genes = self._return_only_valid(druggable_genes, not_predicted_genes)
        valid_druggable_genes = self._return_only_valid(druggable_genes, all_genes)

        assert len(valid_druggable_genes) == len(positive_druggable_genes) + len(predicted_druggable_genes) + len(noncandidate_druggable_genes)

        df["N Druggable"] = [len(positive_druggable_genes), len(predicted_druggable_genes), len(noncandidate_druggable_genes)]

        self.pp_table.add("Druggable", valid_druggable_genes, True, False)

        array = self.make_contingency_table(all_genes, positive_genes, valid_druggable_genes)
        total_druggable = []
        mendelian_druggable_enrichment_result = fisher_exact(array)
        total_druggable.append(mendelian_druggable_enrichment_result)
        logger.info("Total of {} druggable genes, {} of them match with our translation table.".format(len(druggable_genes), len(valid_druggable_genes)))
        logger.info("Found {} druggable genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
            len(druggable_genes.intersection(positive_genes)), len(positive_genes), mendelian_druggable_enrichment_result[1], round(mendelian_druggable_enrichment_result[0], 3), len(unknown_druggable_genes), len(unknown_genes)))

        array = self.make_contingency_table(unknown_genes, predicted_genes, unknown_druggable_genes)
        druggable_enrichment_result = fisher_exact(array)
        total_druggable.append(druggable_enrichment_result)

        logger.info("Fishers Exact Test for Druggable Genes among Predicted Genes. p: {:.2e}, OR: {}".format(druggable_enrichment_result[1], round(druggable_enrichment_result[0], 3)))
        logger.info("Druggable Genes Confusion Matrix:\n" + str(array))

        df["OR Dr"] = [mendelian_druggable_enrichment_result[0], druggable_enrichment_result[0], np.nan]
        df["pval Dr unadjusted"] = [mendelian_druggable_enrichment_result[1], druggable_enrichment_result[1], np.nan]

        # Now we subtract the already known drug targets
        drug_targets = self.get_drugtargets()
        
        all_genes = all_genes - drug_targets
        unknown_genes = unknown_genes - drug_targets
        positive_genes = positive_genes - drug_targets
        druggable_genes = druggable_genes - drug_targets
        valid_druggable_genes = valid_druggable_genes - drug_targets
        unknown_druggable_genes = unknown_druggable_genes - drug_targets
        predicted_genes = predicted_genes - drug_targets
        not_predicted_genes = not_predicted_genes - drug_targets

        assert len(all_genes) == len(positive_genes) + len(predicted_genes) + len(not_predicted_genes)

        df["N Non-Drugtarget"] = [len(positive_genes), len(predicted_genes), len(not_predicted_genes)]

        positive_druggable_genes = positive_druggable_genes - drug_targets
        predicted_druggable_genes = predicted_druggable_genes - drug_targets
        noncandidate_druggable_genes = predicted_druggable_genes  - drug_targets

        df["N Druggable among Non-Drugargets"] = [len(positive_druggable_genes), len(predicted_druggable_genes), len(noncandidate_druggable_genes)]

        leftover_druggable = []
        array = self.make_contingency_table(all_genes, positive_genes, valid_druggable_genes)
        mendelian_druggable_enrichment_result = fisher_exact(array)
        leftover_druggable.append(mendelian_druggable_enrichment_result)
        
        logger.info("Total of {} druggable genes which are not yet Drug Targets, {} of them match with our translation table.".format(len(druggable_genes), len(valid_druggable_genes)))
        logger.info("Found {} druggable non drug target genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
            len(druggable_genes.intersection(positive_genes)), len(positive_genes), mendelian_druggable_enrichment_result[1], round(mendelian_druggable_enrichment_result[0], 3), len(unknown_druggable_genes), len(unknown_genes)))

        predicted_genes = set(self.outer_result[0].keys())

        array = self.make_contingency_table(unknown_genes, predicted_genes, unknown_druggable_genes)
        druggable_enrichment_result = fisher_exact(array)
        leftover_druggable.append(druggable_enrichment_result)

        logger.info("Fishers Exact Test for Druggable Non Drug Target Genes among Predicted Genes. p: {:.2e}, OR: {}".format(druggable_enrichment_result[1], round(druggable_enrichment_result[0], 3)))
        logger.info("Druggable Genes Confusion Matrix:\n" + str(array))

        df["OR Dr-"] = [mendelian_druggable_enrichment_result[0], druggable_enrichment_result[0], np.nan]
        df["pval Dr- unadjusted"] = [mendelian_druggable_enrichment_result[1], druggable_enrichment_result[1], np.nan]

        return total_druggable, leftover_druggable, df

    def mouseKO(self, results_path=None, plot=False, save=True):
        """ Takes the results of the outer crossvalidation and analyzes if there is an enrichment of mouse KO genes among the predicted genes.
            Genes that have not been tested in mouse KO experiments at all have been excluded.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                tuple(ResultA, ResultB, ArrayA, ArrayB): Returns a tuple of various results.
                    ResultA is the enrichment of mouse KO genes in positively labeled genes, ResultB is the enrichment in the candidates.
                    ArrayA is the contingency table of the mouse KO genes with positively Labeled Genes, ArrayB is the contingency table of mouse KO genes with candidate genes.
        """

        logger = setup_logger(*self.logger_args)

        if self.outer_result is None:
            logger.warning("Mouse KO Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        from scipy.stats import fisher_exact

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)
        background_ko_genes = set(self.get_mouse_knockout_background())
        valid_background_ko_genes = self._return_only_valid(background_ko_genes, all_genes)
        unknown_background_ko_genes = self._return_only_valid(valid_background_ko_genes, unknown_genes)
        self.pp_table.add("Included in Mouse KO", valid_background_ko_genes, True, False)
        
        try:
            ko_genes = set(self.get_mouse_knockout_genes())
        except KeyError:
            logger.warning("Phenotype {} not registered for mouse KO analysis.")
            return
        
        mendelian_array = self.make_contingency_table(valid_background_ko_genes, positive_genes, ko_genes.intersection(all_genes))
        mendelian_ko_enrichment_result = fisher_exact(mendelian_array)

        valid_ko_genes = self._return_only_valid(ko_genes, all_genes)
        self.pp_table.add("Is Mouse KO", valid_ko_genes, True, False)
        unknown_ko_genes = self._return_only_valid(ko_genes, unknown_background_ko_genes)

        logger.info("Total of {} Mouse KO genes, {} of them match with our translation table.".format(len(ko_genes), len(ko_genes.intersection(all_genes))))
        logger.info("Found {} Mouse KO genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
            len(ko_genes.intersection(positive_genes)), len(positive_genes), mendelian_ko_enrichment_result[1], round(mendelian_ko_enrichment_result[0], 3), len(unknown_ko_genes), len(unknown_background_ko_genes)))

        predicted_genes = set(self.outer_result[0].keys())

        array = self.make_contingency_table(unknown_background_ko_genes, predicted_genes, unknown_ko_genes)

        ko_enrichment_result = fisher_exact(array)

        logger.info("Fishers Exact Test for mouse KO Genes among Predicted Genes. p: {:.2e}, OR: {}".format(ko_enrichment_result[1], round(ko_enrichment_result[0], 3)))
        logger.info("Mouse KO Confusion Matrix:\n" + str(array))

        return mendelian_ko_enrichment_result, ko_enrichment_result, mendelian_array, array

    def lof_intolerance(self, results_path=None, plot=True, save=False):
        """ Takes the results of the outer crossvalidation and analyzes if there is an enrichment of loss of function and missense mutation intolerant genes among the predicted genes.
            Genes for which we have no LoF or missense information have been excluded.

            Args:
                results_path (str): The path to a resultsfile so the positive labels can be extracted. 
                    This is not necessary if the task :obj:`overlap_analysis` has been run before, then the results paths are already known to the postprocessor.
                plot (bool): If plots should be produced. If True, then the plots are placed in :obj:`config.pp.plot_dir`.
                save (bool): If results should be saved. If True, then the results are placed in the plots in :obj:`config.pp.save_dir`.

            Returns:
                tuple(list[ResultA, ArrayA, ResultB, ArrayB], list[ResultC, ResultD]): Returns a tuple of various results.
                    ResultA is the enrichment of genes with pLI > 0.8 in positively labeled genes, ResultB is the enrichment in the candidates.
                    Array A and ArrayB are the contingency tables for ResultA and ResultB.
                    ResultC is the result of a tukey's HSD test for LoF mutation intolerance among positives, candidates and noncandidates.
                    ResultD is the result of a tukey's HSD test for Missense mutation intolerance among positives, candidates and noncandidates.
        """

        logger = setup_logger(*self.logger_args)
        if self.outer_result is None:
            logger.warning("LoF Intolerance Analysis is requested but results of outer overlap analysis are missing. Skipping it.")
            return

        from scipy.stats import fisher_exact
        from scipy.stats import f_oneway
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        import matplotlib.pyplot as plt

        if "Total" in self.outer_result[0].keys():
            self.outer_result[0].pop("Total")

        unknown_genes, all_genes, positive_genes = self.get_unknown_genes(results_path)

        pli_table = self.get_pli_table()
        pli_genes = set(pli_table["gene"][pli_table["pLI"] > 0.9].tolist())
        all_pli_genes = set(pli_table["gene"].tolist())

        array_mendelian = self.make_contingency_table(all_genes, positive_genes, pli_genes.intersection(all_genes))
        pli_enrichment_result_mendelian = fisher_exact(array_mendelian)

        valid_pli_genes = self._return_only_valid(pli_genes, all_genes)
        self.pp_table.add("pLI>0.9", valid_pli_genes, True, False)

        unknown_pli_genes = self._return_only_valid(pli_genes, unknown_genes)

        logger.info("Total of {} genes with significant LoF Intolerance, {} of them match with our translation table.".format(len(pli_genes), len(pli_genes.intersection(all_genes))))
        logger.info("Found {} LoF Intolerance genes among the {} known positive genes (p: {:.2e}, OR: {}), leaving {} in {} Unknowns".format(
            len(pli_genes.intersection(positive_genes)), len(positive_genes), pli_enrichment_result_mendelian[1], round(pli_enrichment_result_mendelian[0], 3), len(unknown_pli_genes), len(unknown_genes)))

        predicted_genes = set(self.outer_result[0].keys())

        array_candidates = self.make_contingency_table(unknown_genes, predicted_genes, unknown_pli_genes)

        pli_enrichment_result_candidates = fisher_exact(array_candidates)

        logger.info("Fishers Exact Test for genes with significant LoF Intolerance among Predicted Genes. p: {:.2e}, OR: {}".format(pli_enrichment_result_candidates[1], round(pli_enrichment_result_candidates[0], 3)))
        logger.info("LoF Intolerance Confusion Matrix:\n" + str(array_candidates))

        tukeys = []

        for column, description in zip(["lof_z", "mis_z"], ["LoF Z Value", "Missense Z Value"]):
            hgnc2value = {hgnc: value for hgnc, value in zip(pli_table["gene"].tolist(), pli_table[column].tolist()) if hgnc in all_genes}

            mendelian = [hgnc2value[hgnc] for hgnc in positive_genes.intersection(all_pli_genes)]
            predicted = [hgnc2value[hgnc] for hgnc in predicted_genes.intersection(all_pli_genes)]
            not_predicted = [hgnc2value[hgnc] for hgnc in (unknown_genes - predicted_genes).intersection(all_pli_genes)]

            result_predicted = f_oneway(mendelian,
                                        predicted,
                                        not_predicted)

            logger.info("ANOVA for {} in Predicted Genes vs Non-Predicted Genes (Unknowns). p: {:.2e}, F: {}".format(description, result_predicted[1], round(result_predicted[0], 3)))

            df = pd.DataFrame({'score': mendelian + predicted + not_predicted,
                               'group': np.repeat(['Mendelian', 'Candidate Gene', 'Noncandidate Gene'], repeats=[len(mendelian), len(predicted), len(not_predicted)])})

            tukey = pairwise_tukeyhsd(endog=df['score'],
                                      groups=df['group'],
                                      alpha=0.05)

            logger.info(tukey.summary())
            
            if plot:
                tukey.plot_simultaneous(comparison_name="Candidate Gene")
                plt.savefig(self.config.name + "_Tukey_" + "_".join(description.split(" ")))
                plt.close()

            tukeys.append(tukey)

        return [pli_enrichment_result_mendelian, array_mendelian, pli_enrichment_result_candidates, array_candidates], tukeys

    def get_unknown_genes(self, results_path=None):
        #translation_table = self.get_translation_table()
        #all_genes = set(translation_table['symbol'].tolist())
        if results_path is None:
            results_path = self.results_paths[0][0]
        positive_genes = set(self.get_true_positives(results_path))
        unlabeled_genes = set(self.get_unlabeled_genes(results_path))
        assert len(positive_genes.intersection(unlabeled_genes)) == 0

        all_genes = set()
        all_genes.update(positive_genes)
        all_genes.update(unlabeled_genes)

        return unlabeled_genes, all_genes, positive_genes

    def _return_only_valid(self, subset: set, super_set: set) -> set:
        """check if each entry in subset are present in super_set and return only those wo are"""

        return subset.intersection(super_set)
        
    @classmethod
    def make_contingency_table(self, full: set, A: set, B: set) -> np.ndarray:
        notA = full - A
        notB = full - B

        A_and_B = A.intersection(B)
        A_not_B = A.intersection(notB)
        not_A_and_B = notA.intersection(B)
        not_A_not_B = notA.intersection(notB)

        return np.array([[len(A_and_B), len(not_A_and_B)], [len(A_not_B), len(not_A_not_B)]])

    def create_if_not_exists(self, path_to_dir):
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

    def get_pli_table(self, path_to_table="data/forweb_cleaned_exac_r03_march16_z_data_pLI.txt") -> tuple:
        return pd.read_csv(path_to_table, header=0, sep="\t")
    
    @property
    def mouse2human(self, path="./data/mgi/HOM_MouseHumanSequence.rpt"):
        import pandas as pd

        if hasattr(self, "_mouse2human"):
            return self._mouse2human

        table = pd.read_csv(path, sep="\t", header=0)

        human = table[table["Common Organism Name"] == "human"]
        mouse = table[table["Common Organism Name"] == "mouse, laboratory"]
        human.index = human["DB Class Key"]
        mouse.index = mouse["DB Class Key"]
        joined = human.join(mouse, how="inner", lsuffix=" human", rsuffix=" mouse")
        mouse2human = {}
        doubles = {}
        for human, mouse in zip(joined["Symbol human"], joined["Symbol mouse"]):
            if mouse in mouse2human.keys():
                mouse2human[mouse].append(human)
                doubles[mouse] = mouse2human[mouse]
            else:
                mouse2human[mouse] = [human]

        self._mouse2human = mouse2human

        return mouse2human

    def get_druggable_genes(self, path_to_table) -> list:
        logger = setup_logger(*self.logger_args)
        logger.info("Reading druggable genes from {}".format(path_to_table))
        return pd.read_csv(path_to_table, sep="\t", header=None).iloc[:, 0].tolist()

    def get_translation_table(self, sep="\t") -> pd.DataFrame:
        logger = setup_logger(*self.logger_args)
        logger.info("Reading translation table from {}".format(self.path_to_translation_table))
        df = pd.read_csv(self.path_to_translation_table, sep=sep, header=0, usecols=[self.hgnc_col, self.entrez_col, self.ensembl_col])
        #df.rename(columns={hgnc_col: self.hgnc_key, entrez_col: self.entrez_key, ensembl_col: self.ensembl_key}, inplace=True)
        return df

    def get_mouse_knockout_genes(self, tag=None, mapping="./data/mgi/query_mapping.yaml") -> list:
        """ 
            Reads the Mouse Knockout genes from mapping file, 
            matches it against the mouse to human homologs (self.mouse2human) and returns the human homologs with a corresponding mouse KO gene 

            Mouse KO genes which do not have human homologs will not be returned.
        """
        import yaml
        logger = setup_logger(*self.logger_args)

        tag = self.config.input.tag if tag is None else tag
        with open(mapping, "r") as file:
            mapping = yaml.load(file, Loader=yaml.SafeLoader)

        path_to_table = None

        for option, value in mapping.items():
            if option.startswith(tag.lower()):
                path_to_table = value["file"]

        if path_to_table is None:
            logger.error("Could not find mouse knockout genes for tag {} in mapping {}".format(tag, mapping))
            return

        logger.info("Reading mouse knockout genes from {}".format(path_to_table))
        mouse_symbols = [entry.split("<")[0] for entry in pd.read_csv(path_to_table, sep="\t", header=0)["Allele Symbol"].tolist()]
        human_homolog_symbols = set()

        for mouse_symbol in mouse_symbols:
            try:
                human_homolog_symbols.update(self.mouse2human[mouse_symbol])
            except KeyError:
                continue

        return human_homolog_symbols

    def get_mouse_knockout_background(self, tag="background"):
        return self.get_mouse_knockout_genes(tag=tag)

    def load_drugtarget_graph(self, path_to_graph):
        import networkx as nx
        import pandas as pd

        logger = setup_logger(*self.logger_args)
        logger.info("Reading compound drug interaction graph from {}".format(path_to_graph))
        df = pd.read_table(path_to_graph, sep="\t", names=["Compound", "edge", "Gene"])

        graph = nx.from_pandas_edgelist(df, source="Compound", target="Gene", edge_attr="edge", create_using=nx.MultiDiGraph)

        return graph

    def get_drugtargets(self, path_to_graph="./data/drkg/cgi.tsv") -> set:
        hgnc2degree = self.get_drugtarget_dict(path_to_graph)
        return set(hgnc2degree.keys())

    def get_drugtarget_dict(self, path_to_graph="./data/drkg/cgi.tsv") -> dict:
        graph = self.load_drugtarget_graph(path_to_graph)
        node2entrez = {node: "".join(node.split("::")[1:]) for node in graph.nodes if node.startswith("Gene")}
        # node2entrez = {value: "".join(value.split("::")[1:]) for value in df["Gene"] if not "".join(value.split("::")[1:]).startswith("drugbank")}
        entrez2degree = {node2entrez[node]: graph.degree[node] for node in node2entrez.keys()}
        translation_table = self.get_translation_table()
        entrez2hgnc = {str(int(translation_table['entrez_id'][i])): translation_table['symbol'][i] for i in range(len(translation_table)) if not np.isnan(translation_table['entrez_id'][i])}
        hgnc2degree = {entrez2hgnc[entrez]: degree for entrez, degree in entrez2degree.items() if entrez in entrez2hgnc.keys()}

        return hgnc2degree

    def make_boxplot(self, noncandidate, candidate, positive, plot=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title('Multiple Samples with Different sizes')
        ax.boxplot([noncandidate, candidate, positive], showfliers=False)
        ax.set_xticklabels(["Non-Candidates", "Candidates", "Mendelians"])
        ax.set_ylabel("Degree")
        fig.tight_layout()

        if plot:
            fig.savefig("CGI_{}.png".format(self.config.name), dpi=400)

        return fig

    def get_true_positives(self, results_path=None):

        table = self._read_results_file(results_path)

        return table["hgnc"][table["truth"] == 1].tolist()

    def get_unlabeled_genes(self, results_path=None):

        table = self._read_results_file(results_path)

        return table["hgnc"][table["truth"] == 0].tolist()

    def overlap_analysis(self, write=True, plot=True):
        """
            Takes the crossval suffix from config and performs an overlap analysis by calling check_overlap() with the given values.
            Stores the results in output_dir if write is set to True, which is default.

            Returns a dictionary that maps the counts to the genes and a list of runs that have been considered for the analysis.
        """
        logger = setup_logger(*self.logger_args)
        logger.info("Starting Overlap Analysis.")
        gene_counters, count_counters, all_pos_pvals, dfs = list(zip(*[self.check_overlap(results_paths, self.config.pp.cutoff_value, self.config.pp.cutoff_type, plot=self.config.pp.plot) for results_paths in self.results_paths]))
        
        if write:
            df = pd.concat(dfs)
            df["Outer Fold"] = np.repeat(range(1, len(dfs) + 1), len(dfs[0])).astype(int)
            df["Inner Overlap Bin"] = df["Inner Overlap Bin"].astype(int)
            columns = [df.columns[-1]] + df.columns[:-1].tolist()
            df = df[columns]
            df.to_csv(os.path.join(self.config.pp.save_dir, str(self.config.name) + "_overlap.tsv"), index=False, sep="\t")

        most_often_predicted_list = []
        handles = []

        for i, (gene_counter, count_counter, pos_pvals) in enumerate(zip(gene_counters, count_counters, all_pos_pvals)):

            labels, values = zip(*sorted(count_counter.items()))

            count2gene = {count: [] for count in labels}

            for gene, count in gene_counter.items():
                try:
                    count2gene[count].append(gene)
                except KeyError:
                    pass

            if self.config.pp.save:
                if len(gene_counters) > 1:
                    handle = os.path.join(self.config.pp.save_dir, str(self.config.name) + self.config.crossval.outer_suffix.format(i) + "_most_often_predicted.txt")
                else:
                    handle = os.path.join(self.config.pp.save_dir, str(self.config.name) + "_most_often_predicted.txt")

                handles.append(handle)

                # most_often_predicted = sorted(count2gene[np.max(list(count2gene.keys()))])
                most_often_predicted, consensus_score = self.get_consensus_genes(count2gene, pos_pvals)
                most_often_predicted_list.append(most_often_predicted)
                logger.info("Consensus Score for Outer Crossval #{}: {}; Returned {} Candidate Genes".format(i, consensus_score, len(most_often_predicted)))
                np.savetxt(handle, most_often_predicted, fmt='%s')

        if len(gene_counters) > 1:
            outer_result = self._count_overlap(most_often_predicted_list)
            total = str(np.sum(list(outer_result[1].values())))
            with open(os.path.join(self.config.pp.save_dir, str(self.config.name) + "outer_results.json"), 'w') as fp:
                json.dump(outer_result, fp, skipkeys=True, indent=2)

            self.outer_result = outer_result
            logger.info("Outer Crossvalidation results in {} candidate genes in total. Results written to {}".format(total, os.path.join(pu.postprocessing_results_path(self.config), str(self.config.name) + "outer_results.json")))
            candidate_genes, consensus_score = list(zip(*outer_result[0].items()))
            self.pp_table.add("Candidate", index=candidate_genes, values=True, remaining=False)
            self.pp_table.add("CS", index=candidate_genes, values=consensus_score, remaining=0)

        logger.info("Finished Overlap Analysis")
        return count2gene,

    def load_outer_results(self, path=None):
        if path is None:
            path = os.path.join(self.config.pp.save_dir, str(self.config.name) + "outer_results.json")
        with open(path, "r") as fp:
            self.outer_result = json.load(fp)

    def get_consensus_genes(self, count2gene, pvals):
        consensus_score = self._find_consensus_score(pvals)
        mask = [np.array(list(count2gene.keys())) >= consensus_score]
        selected_keys = np.array(list(count2gene.keys()))[tuple(mask)]
        return sorted([gene for key in selected_keys for gene in count2gene[key]]), consensus_score

    def _find_consensus_score(self, pvals: list) -> int:
        """from 1 to n, finds in which bin the positive values are significantly enriched for the first time"""
        if type(self.consensus) == int:
            return self.consensus
        elif self.consensus == "bottom_up":
            for i, pval in enumerate(pvals):
                if pval < 0.05:
                    return i + 1
        elif self.consensus == "top_down":
            for i, pval in enumerate(pvals[::-1]):
                if pval > 0.05:
                    return len(pvals) - i + 1

    def _read_results_file(self, results_file=None):
        """Reads and returns the specified results file.
           If no results file is specified, reads the first.
           If a list is passed as results_file, the first element is read."""

        if results_file is None:
            results_file = self.results_paths[0]
        if type(results_file) == list:
            results_file = results_file[0]
        return pd.read_csv(results_file, header=0, index_col=0, sep="\t")

    def check_overlap(self, results_paths: list, cutoff_value, cutoff_type: str, plot=True):
        """Checks the overlap of multiple runs and returns them
        
            Args:
                results_paths (list): Paths to the results files that should be compared for overlap (i.e. all results files from one outer cv run)
                cutoff_value (int/float): Value which is depends on :obj:`cutoff_type` .
                cutoff_type (str): Type of cutoff that should be applied to find candidates. See :doc:`api` for possible values.
                plot (bool): If we overlap bins should be plotted.

            Returns:
                tuple([..., pd.DataFrame]): Multiple results, most of which are summarized in the DataFrame at the end (i.e. tuple[-1]) 
                
        """

        logger = setup_logger(*self.logger_args)

        results_tables = [self._read_results_file(path) for path in results_paths]

        count_counters = {}
        gene_counters = {}
        total_gene_set_sizes = {}

        for criterion in ["Unknown", "Test Positive"]:
            if self.config.pp.cutoff_type in ["top", "bottom"]:
                if criterion == "Unknown":
                    sorted_genes = [table[table["truth"] == 0].sort_values(by="probability", axis=0, ascending=False)["hgnc"].tolist() for table in results_tables]
                elif criterion == "Test Positive":
                    sorted_genes = [table[(table["truth"] == 1) & (table["test"] == 1)].sort_values(by="probability", axis=0, ascending=False)["hgnc"].tolist() for table in results_tables]
                kept_genes = [genes[:self.config.pp.cutoff_value] if cutoff_type == "top" else genes[-self.config.pp.cutoff_value:] for genes in sorted_genes]
            elif self.config.pp.cutoff_type == "split":
                if criterion == "Unknown":
                    eligible_genes = [table["hgnc"][table["truth"] == 0].to_numpy() for table in results_tables]
                    kept_genes = [table["hgnc"][(table["probability"] > self.config.pp.cutoff_value) & (table["truth"] == 0)].tolist() for table in results_tables]
                elif criterion == "Test Positive":
                    eligible_genes = [table["hgnc"][(table["truth"] == 1) & (table["test"] == 1)].to_numpy() for table in results_tables]
                    kept_genes = [table["hgnc"][(table["probability"] > self.config.pp.cutoff_value) & (table["truth"] == 1) & (table["test"] == 1)].tolist() for table in results_tables]
            else:
                logger.error("Cutoff Type {} not in implemented cutoff types 'top', 'bottom' or 'split'".format(self.config.pp.cutoff_type))
                raise ValueError

            total_gene_set_sizes[criterion] = [len(eligible_gene_set) for eligible_gene_set in eligible_genes]

            #random_genes = {iteration: [total_genes[np.array(random.sample(range(len(total_genes)), len(nonrandom_genes)))].tolist() if len(nonrandom_genes) > 0 else [] for total_genes, nonrandom_genes in zip(eligible_genes, kept_genes)] for iteration in range(self.num_runs_for_random_experiments)}


            gene_counter, count_counter = self._count_overlap(kept_genes)
            mean_counter, sd_counter = self.get_random_overlap(eligible_genes, kept_genes, algorithm="descriptive")
            #mean_counter, sd_counter = self._count_overlap(random_genes, get_mean_sd=True)

            gene_counters.update({criterion: gene_counter})
            count_counters.update({criterion: count_counter})
            count_counters.update({criterion + " Random Mean": mean_counter})
            count_counters.update({criterion + " Random SD": sd_counter})

        if plot:
            plot_title = "Overlap in predicted disease genes between {} runs, type {} {}".format(len(results_paths), cutoff_type, cutoff_value)
            plot_name = self.longest_common_string(results_paths[0].split("/")[-1], results_paths[1].split("/")[-1]) + "_overlap.svg"
            pos_test_pvals, df = self.plot_overlap(count_counters, total_gene_set_sizes, plot_title, plot_name)

        return gene_counters["Unknown"], count_counters["Unknown"], pos_test_pvals, df

    def get_random_overlap(self, eligible_genes, kept_genes, algorithm="descriptive", n_models = None):
        """
            Gets the same number of random genes as in kept_genes out of eligible genes and repeats this procedure self.num_runs_for_random_experiments times to get mean and standard deviation of overlaps

            if algorithm="descriptive", then we sample from an actual list of gene symbols. If algorithm="fast", we recreate the sampling as a bernoulli experiment in scipy, which is much faster.
        """
        import scipy.stats as stats
        if algorithm == "descriptive":
            random_genes = {iteration: [total_genes[np.array(random.sample(range(len(total_genes)), len(nonrandom_genes)))].tolist() if len(nonrandom_genes) > 0 else [] for total_genes, nonrandom_genes in zip(eligible_genes, kept_genes)] for iteration in range(self.num_runs_for_random_experiments)}
            mean_counter, sd_counter = self._count_overlap(random_genes, get_mean_sd=True)
        elif algorithm == "fast":
            n_models = self.config.crossval.n_folds if n_models is None else n_models
            n_drawings = self.num_runs_for_random_experiments

            b = []
            results = []
            for eligible_genes_one_fold, kept_genes_one_fold in zip(eligible_genes, kept_genes):
                results.append(stats.binom.rvs(1, len(kept_genes_one_fold)/len(eligible_genes_one_fold), size=(len(eligible_genes_one_fold), n_drawings)))
                
            results = np.array(results)

            result = results.sum(axis=0)
                
            _, counts = zip(*[np.unique(result[:,i], return_counts=True) for i in range(result.shape[1])])

                

            for i in range(len(counts)):
                if len(counts[i]) != n_models + 1:
                    b.append(np.pad(counts[i], (0, n_models + 1 - len(counts[i])), 'constant', constant_values = 0))
                else:
                    b.append(counts[i])
                        
            b = np.array(b)

            mean_counter = {i + 1: mean.item() for i, mean in enumerate(b.mean(axis=0)[1:])}
            sd_counter = {i + 1: mean.item() for i, mean in enumerate(b.std(axis=0)[1:])}
        else:
            raise ValueError("'algorithm' keyword must be either 'descriptive' or 'fast'")
        return  mean_counter, sd_counter

    def _count_overlap(self, list_of_sets, get_mean_sd: bool = False):
        ''' Takes a list of gene sets and returns counts how often each gene occurs in each of the sets and how often each of the counts occurs in the total list.
        optional parameter get_mean_sd means that the input value is a dict of list of sets containing multiple iterations of random experiments and mean and sd of counts is returned instead'''

        if get_mean_sd:
            count_counters = [self._count_overlap(list)[1] for _, list in list_of_sets.items()]
            aggregated_count_counters = {i: [] for i in range(1, len(count_counters[0].keys()) + 1)}

            for i in aggregated_count_counters.keys():
                aggregated_count_counters[i].append(np.array([count_counter[i] for count_counter in count_counters]))

            mean_counters = {count: np.mean(values) for count, values in aggregated_count_counters.items()}
            sd_counters = {count: np.std(values) for count, values in aggregated_count_counters.items()}

            return mean_counters, sd_counters

        else:
            flat_list = [item for sublist in list_of_sets for item in sublist]

            if len(flat_list) == 0:
                raise ValueError("No genes match criteria of type {} and value {}.".format(self.config.pp.cutoff_type, self.config.pp.cutoff_value))

            gene_counter = Counter(flat_list)

            labels, values = zip(*gene_counter.items())

            values = np.array(values)
            illegal_values_mask = values > len(list_of_sets)
            illegal_values_idx = np.nonzero(illegal_values_mask)
            illegal_genes = np.array(labels)[illegal_values_idx]

            if len(illegal_genes) > 0 and len(illegal_genes) < 10:
                logging.warning("Encountered illegal counts for the genes {}: {}, removing them.".format(illegal_genes, values[illegal_values_idx]))
                values = values[~illegal_values_mask]
            elif len(illegal_genes) > 10:
                logging.warning("Encountered illegal counts for 10 or more genes, removing them.")
                values = values[~illegal_values_mask]

            count_counter = Counter(values.tolist())

            for i in range(1, len(list_of_sets) + 1):
                # check if a gene count doesnt occur (i.e. no gene has been predicted exactly 8 times)
                if i not in count_counter.keys():
                    # and add a 0 for that count
                    count_counter[i] = 0

            return gene_counter, count_counter

    def plot_overlap(self, count_counters, total_gene_set_sizes, plot_title, plot_name, percentage=True, vector=True) -> list:
        from scipy.stats import t
        from speos.scripts.utils import fdr

        red = '#fd151b'
        orange = '#ffb30f'
        darkblue = '#01295f'
        lightblue = '#437f97'
        background_grey = '#e2e3e4'

        unknown_labels, unknown_values = zip(*sorted(count_counters["Unknown"].items()))
        _, unknown_random_mean_values = zip(*sorted(count_counters["Unknown Random Mean"].items()))
        _, unknown_random_sd_values = zip(*sorted(count_counters["Unknown Random SD"].items()))
        pos_val_labels, pos_val_values = zip(*sorted(count_counters["Test Positive"].items()))
        _, pos_val_random_mean_values = zip(*sorted(count_counters["Test Positive Random Mean"].items()))
        _, pos_val_random_sd_values = zip(*sorted(count_counters["Test Positive Random SD"].items()))
        
        num_unknown = total_gene_set_sizes["Unknown"]
        num_test_pos = total_gene_set_sizes["Test Positive"]

        # if there are no predicted genes and no random genes in a bin, just count it as significant
        pvals_unknown = [t.sf(unknown_value, self.num_runs_for_random_experiments, random_mean, random_sd + 1e-16) if random_mean > 0.1 else 0 for unknown_value, random_mean, random_sd in zip(unknown_values, unknown_random_mean_values, unknown_random_sd_values)]
        pvals_pos_val = [t.sf(pos_val_value, self.num_runs_for_random_experiments, random_mean, random_sd + 1e-16) if random_mean > 0.1 else 0 for pos_val_value, random_mean, random_sd in zip(pos_val_values, pos_val_random_mean_values, pos_val_random_sd_values)]

        pvals_adjusted = fdr([*pvals_unknown, *pvals_pos_val])
        pvals_adj_unknown = pvals_adjusted[:len(pvals_unknown)]
        pvals_adj_pos_val = pvals_adjusted[len(pvals_unknown):]

        df = pd.DataFrame(data=np.vstack((range(1, len(unknown_values) + 1), unknown_values, unknown_random_mean_values, unknown_random_sd_values, pvals_unknown, pvals_adj_unknown,
                                                                             pos_val_values, pos_val_random_mean_values, pos_val_random_sd_values, pvals_pos_val, pvals_adj_pos_val)).transpose(),
                          columns=["Inner Overlap Bin", "Unlabeled Count", "Unlabeled Random Mean", "Unlabeled Random SD", "Unlabeled pval", "Unlabeled pval adjusted (FDR)",
                                                        "Pos Test Count", "Pos Test Random Mean", "Pos Test Random SD", "Pos Test pval", "Pos Test pval adjusted (FDR)"])

        indexes = np.arange(max(unknown_labels))
        width = 0.22

        fig, ax = plt.subplots()

        if percentage:
            unknown_values = [(unknown_value / total_num_unknown) * 100 for unknown_value, total_num_unknown in zip(unknown_values, num_unknown)]
            unknown_random_mean_values = [(unknown_random_mean_value / total_num_unknown) * 100 for unknown_random_mean_value, total_num_unknown in zip(unknown_random_mean_values, num_unknown)]
            unknown_random_sd_values =[(unknown_random_sd_value / total_num_unknown) * 100 for unknown_random_sd_value, total_num_unknown in zip(unknown_random_sd_values, num_unknown)]
            pos_val_values = [(pos_val_value / total_num_test_pos) * 100 for pos_val_value, total_num_test_pos in zip(pos_val_values, num_test_pos)]
            pos_val_random_mean_values = [(pos_val_random_mean_value / total_num_test_pos) * 100 for pos_val_random_mean_value, total_num_test_pos in zip(pos_val_random_mean_values, num_test_pos)]
            pos_val_random_sd_values = [(pos_val_random_sd_value / total_num_test_pos) * 100 for pos_val_random_sd_value, total_num_test_pos in zip(pos_val_random_sd_values, num_test_pos)]

        if percentage:
            ax.set_ylabel('% of Genes')
            global_max = np.max([*unknown_values, *unknown_random_mean_values, *pos_val_values, *pos_val_random_mean_values])
            cutoff_at = global_max + 5
        else:
            ax.set_ylabel('# Genes')
            cutoff_at = 1000

        _ = ax.bar([index for index in indexes if index % 2 == 1], cutoff_at, 1, color=background_grey, zorder=-5)
        ax.yaxis.grid(color="lightgrey", linestyle=':', linewidth=1, zorder=-1)
        rects1 = ax.bar(indexes - width * 3 / 2, unknown_values, width, label='Unknown (n={})'.format(num_unknown[0]).format(), color=darkblue, zorder = 5)
        rects2 = ax.bar(indexes - width * 1 / 2, unknown_random_mean_values, width, yerr=unknown_random_sd_values, label='Unknown Rand Ctrl', color=lightblue, zorder = 5)
        rects3 = ax.bar(indexes + width * 1 / 2, pos_val_values, width, label='Test Positive (n={})'.format(num_test_pos[0]), color=red, zorder = 5)
        rects4 = ax.bar(indexes + width * 3 / 2, pos_val_random_mean_values, width, yerr=pos_val_random_sd_values, label='Test Positive Rand Ctrl', color=orange, zorder = 5)
        
        #ax.set_title(plot_title)
        ax.set_xticks(indexes, unknown_labels)
        
        ax.set_ylim(0, cutoff_at)
        ax.legend()

        for rects in [rects1, rects2, rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                if not percentage:
                    plt.text(rect.get_x() + rect.get_width() / 2.25, height + (0.003*cutoff_at) if height < cutoff_at else cutoff_at - 30, round(height, 1), ha='center', va='bottom')
        for i, rect in enumerate(rects1):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.25, height + (0.02*cutoff_at) if height < cutoff_at else cutoff_at - 10, "*" if pvals_adj_unknown[i] < 0.05 else "", ha='center', va='bottom', color="grey")
        for i, rect in enumerate(rects3):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.25, height + (0.02*cutoff_at) if height < cutoff_at else cutoff_at - 10, "*" if pvals_adj_pos_val[i] < 0.05 else "", ha='center', va='bottom')

        ax.set_xlabel("Predicted in # of models")
        fig.set_size_inches(7, 4)
        fig.set_dpi(300)
        fig.tight_layout()
        logger = setup_logger(*self.logger_args)
        logger.info("Plotting overlap plot to {}".format(plot_name))
        plt.savefig(plot_name)
        plt.close()

        return pvals_adj_pos_val, df

    @property
    def results_paths(self):
        import re
        output_dir = self.config.pp.save_dir
        suffix = self.config.crossval.suffix
        outer_suffix = self.config.crossval.outer_suffix

        regex = re.compile(str(self.config.name) + outer_suffix.format("[0-9]+") + suffix.format("[0-9]+") + ".tsv")

        filenames = [filename for _, _, filename in os.walk(output_dir)][0]

        results_paths = [os.path.join(output_dir, regex.match(filename).group(0)) for filename in filenames if regex.match(filename)]

        regex_template = ".*" + str(self.config.name) + outer_suffix + "_fold.*.tsv"

        results_paths_new = [[re.compile(regex_template.format(i)).match(path).group(0) for path in results_paths if re.compile(regex_template.format(i)).match(path)] for i in range(self.config.crossval.n_folds + 1)]

        return results_paths_new

    def longest_common_string(self, string1, string2):
        longest_common_string = ""

        # make sure string1 is the shorter string
        if len(string1) > len(string2):
            string1, string2 = string2, string1

        for char in string1:
            if string2.startswith(char):
                longest_common_string += char
                string2 = string2[1:]
            else:
                break

        return longest_common_string
