from speos.visualization.graphdiagnostic.graphdiagnostic import GraphDiagnostic
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.preprocessor import PreProcessor
from speos.utils.config import Config
import json
import os
import argparse
import numpy as np
from scipy.stats import fisher_exact

parser = argparse.ArgumentParser(description='Plot statistical properties of results')

parser.add_argument('--config', "-c", type=str,
                    help='Path to the config that should be used for the run.')
parser.add_argument('--cutoff', "-k", type=int, default=1,
                    help='cutoff convergence score.')
parser.add_argument('--comparison', "-d", type=str, default=">=",
                    help='comparison operacor for cutoff, either ">=" or "==".')
parser.add_argument('--handle', type=int, default=0,
                    help='if --comparison "==", how to handle all genes with CS != cutoff. 0: exclude none, 1: exclude higher CS, 2: exclude all with CS > 0')

args = parser.parse_args()


class ResultsDiagnostic(GraphDiagnostic):
    """Class that checks the results of a crossvalidation run for correlation with graph properties such as homophily and node degree"""
    def __init__(self, config=None):
        self.config = Config() if config is None else config
        self.gwasmapper = GWASMapper()
        self.adjacencymapper = AdjacencyMapper(config.input.adjacency_mappings, blacklist=self.config.input.adjacency_blacklist)

        mappings = GWASMapper().get_mappings(
            config.input.tag, fields=config.input.field)

        tag = "" if config.input.adjacency == "all" else config.input.adjacency
        adjacencies = AdjacencyMapper(config.input.adjacency_mappings, blacklist=self.config.input.adjacency_blacklist).get_mappings(tag, fields=config.input.adjacency_field)

        self.prepro = PreProcessor(self.config, mappings, adjacencies)
        self.G = self.prepro.get_graph()

        with open(os.path.join(self.config.pp.save_dir, str(self.config.name) + "outer_results.json"), "r") as file:
            self.results = json.load(file)[0]

    def get_candidates(self, n = None, comparison = None):
        global args
        if n is None:
            n = args.cutoff 
        if comparison is None:
            comparison = args.comparison
        if comparison == "==":
            assert args.handle in [0,1,2], ("Can only use values 0, 1 or 2 for handle argument")

        all_keys = list(self.prepro.hgnc2id.keys())

        mighty_keys = []
        for key in all_keys:
            if key in self.results.keys():
                if comparison == "==":
                    if self.results[key] == args.cutoff:
                        mighty_keys.append(key)
                elif comparison == ">=":
                    if self.results[key] >= args.cutoff:
                        mighty_keys.append(key)
        return mighty_keys
        

    def get_degrees(self, graph=None):
        if graph is None:
            graph = self.G
        all_keys = list(self.prepro.hgnc2id.keys())
        connected_and_positive = 0
        connected_and_negative = 0
        disconnected_and_positive = 0
        disconnected_and_negative = 0
        bins = {convergence_score: [] for convergence_score in range(12)}
        global args
        candidates = set(self.get_candidates())
        for key in all_keys:
            degree = graph.degree[self.prepro.hgnc2id[key]]
            if key in candidates:
                bins[self.results[key]].append(degree)
                if degree > 0:
                    connected_and_positive += 1
                else: 
                    disconnected_and_positive += 1
            else:
                bins[0].append(degree)
                if degree > 0:
                    connected_and_negative += 1
                else: 
                    disconnected_and_negative += 1

        array = np.array(((connected_and_positive, connected_and_negative), (disconnected_and_positive, disconnected_and_negative)), dtype=np.uint16)

        test_result = fisher_exact(array)

        print("Fishers Exact Test for Connected Genes among Predicted Genes. p: {:.2e}, OR: {}".format(test_result[1], round(test_result[0], 3)))
        print(array)

        return bins

    def get_homophily_per_cs(self, graph=None):
        import numpy as np
        if graph is None:
            graph = self.G
        all_keys = list(self.prepro.hgnc2id.keys())
        bins_absolute = {convergence_score: [] for convergence_score in range(12)}
        bins_relative = {convergence_score: [] for convergence_score in range(12)}
        for key in all_keys:
            neighbors = list(graph.neighbors(self.prepro.hgnc2id[key]))
            if len(neighbors) > 0:
                num_pos = np.sum([graph.nodes[n]["y"] == 1 for n in neighbors])
                frac_pos = num_pos / len(neighbors)
                if key in self.results.keys():
                    bins_absolute[self.results[key]].append(num_pos)
                    bins_relative[self.results[key]].append(frac_pos)
                else:
                    bins_absolute[0].append(num_pos)
                    bins_relative[0].append(frac_pos)

        return bins_absolute, bins_relative

    def plot_homophily_per_cs(self, graph=None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr
        absolute, relative = self.get_homophily_per_cs()
        for i, (method, bins) in enumerate(zip(["Absolute", "Relative"], (absolute, relative))):
            fig, ax = plt.subplots()
            x = [[bin_number] * len(bin_content) for bin_number, bin_content in bins.items()]
            x = [flat_list for ragged_list in x for flat_list in ragged_list]
            y = [flat_list for ragged_list in bins.values() for flat_list in ragged_list]
            ax = sns.violinplot(x=x, y=y, ax=ax, cut=0)
            ax = sns.regplot(x=x, y=y, ax=ax, scatter=False)
            ax.set_xlabel("Convergence Threshold")
            if method == "Absolute":
                ax.set_ylabel("Number of Neighborhood Positives")
            else:
                ax.set_ylabel("Fraction of Neighborhood Positives")
            r, p = spearmanr(x, y)
            ax.text(1, np.max(y) - (0.05 * (np.max(y) - np.min(y))), "r={:.2f}, p={:.3f}".format(r, p))
            plt.savefig("Results_homohpily_{}_{}.png".format(method, self.config.name), dpi=450)

    def plot_degrees(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr
        fig, ax = plt.subplots()
        degrees = self.get_degrees()
        x = [[bin_number] * len(bin_content) for bin_number, bin_content in degrees.items()]
        x = [flat_list for ragged_list in x for flat_list in ragged_list]
        y = np.log10([flat_list + 1 for ragged_list in degrees.values() for flat_list in ragged_list])
        ax = sns.violinplot(x=x, y=y, ax=ax, cut=0)
        ax = sns.regplot(x=x, y=y, ax=ax, scatter=False)
        ax.set_xlabel("Convergence Threshold")
        ax.set_ylabel("log(degree + 1)")
        r, p = spearmanr(x, y)
        ax.text(1, np.max(y) - (0.05 * (np.max(y) - np.min(y))), "r={:.2f}, p={:.3f}".format(r, p))
        plt.savefig("Results_degrees_{}.png".format(self.config.name), dpi=450)

    def plot_results_homophily(self):
        candidates = set(self.get_candidates())
        global args
        graph = self.G.copy()
        for node in graph.nodes(data=True):
            if node[1]["hgnc"] in candidates:
                node[1]["y"] = 1
            else:
                node[1]["y"] = 0

        fig, ax = super().check_homophily(graph = graph) 
        
        fig.savefig("homophily_results_{}_{}_{}_handle{}.png".format(self.config.name, args.comparison, args.cutoff, args.handle), dpi=350)



config = Config()
config.parse_yaml(args.config)
diagnostic = ResultsDiagnostic(config=config)

#diagnostic.plot_degrees()
#diagnostic.plot_homophily()
diagnostic.plot_results_homophily()