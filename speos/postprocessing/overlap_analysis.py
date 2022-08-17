from postprocessor import PostProcessor
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import Counter
import logging

def main():
    """checks the overelap of a bunch of runs and plots it"""

    # get bulk resultsfiles

    handle= 'Immune_Dysregulation_CV_2nd_split.*\.tsv'
    cutoff_value = 0.7
    cutoff_type = "split"

    rootdir = "./results/"
    regex = re.compile(handle)
    results_paths = []

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                results_paths.append(rootdir + file)

    postprocessor = PostProcessor()

    gene_counter, count_counter = postprocessor.check_overlap(results_paths, cutoff_value, cutoff_type, plot=True)

    labels, values = zip(*sorted(count_counter.items()))

    count2gene = {count: [] for count in labels}

    for gene, count in gene_counter.items():
        try:
            count2gene[count].append(gene)
        except KeyError:
            pass

    np.savetxt(rootdir + handle.split(".")[0] + "_most_often_predicted.txt", sorted(count2gene[np.max(list(count2gene.keys()))]), fmt='%s')

if __name__ == "__main__":
    main()