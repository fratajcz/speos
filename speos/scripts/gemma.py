import pandas as pd
import numpy as np
from scipy.stats import chi2 
import sys
from utils import fdr


def get_fisher_pval(df,alpha=0.05):
    return np.exp(-0.5*chi2.isf(alpha,2*df))


#exp_results_raw = pd.read_csv("results/gemma/gemma_pvals.tsv", sep="\t",header=0)
#exp_results_raw = pd.read_csv("results/gemma/ibd_pvals.tsv", sep="\t",header=0)
#exp_results_raw = pd.read_csv("results/gemma/random_ibd_pvals.tsv", sep="\t",header=0)
#exp_results_raw = pd.read_csv("results/gemma/random_cardiovascular_pvals.tsv", sep="\t",header=0)
exp_results_raw = pd.read_csv("results/gemma/cardiovascular_pvals.tsv", sep="\t",header=0)
exp_results = pd.DataFrame(exp_results_raw["Gene"]).drop(0)

exp_results_diseases = exp_results_raw.filter(regex="'disease vs reference subject role", axis=1).drop(0)
#exp_results_diseases =  exp_results_raw.drop(0).drop(["Gene","Meta p-value"],axis= "columns")


meta_pvals = [] 
meta_statistics = []
fisher_pvals = []
fisher_decision_borders = []
new_pvals = []

for i, row in exp_results_diseases.iterrows():
    values = fdr(row.values[~np.isnan(row.values)])
    meta_pvals.append(np.prod(values))
    fisher_pvals.append(get_fisher_pval(len(values)))
    meta_statistics.append(-2 * np.sum(np.log(values)))
    fisher_decision_borders.append(chi2.isf(0.05,2*len(values)))

    new_pvals.append(np.min((2*(1 - chi2.cdf(abs(meta_statistics[-1]), 2*len(values))),1)))

new_pvals_fdr = fdr(np.array(new_pvals))
significant_adjusted = np.sum(new_pvals_fdr < 0.05)
significant_unadjusted = np.sum(np.array(new_pvals) < 0.05)
print("Found significant differential expression in {} out of {} genes ({} without pvalue adjustment)".format(significant_adjusted,len(exp_results_diseases),significant_unadjusted))