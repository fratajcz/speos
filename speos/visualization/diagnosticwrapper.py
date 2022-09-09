from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.preprocessor import PreProcessor
from speos.utils.config import Config
from speos.visualization.graphdiagnostic.graphdiagnostic import GraphDiagnostic

import matplotlib.pyplot as plt
import numpy as np
import os


class GraphDiagnosticWrapper:
    def __init__(self, preprocessor=None, config=None, gwasmapper=None, adjacencymapper=None, phenotype_tag="", adjacency_tag="", phenotype_fields="name", adjacency_fields="name", merge=False):

        self.config = Config() if config is None else config
        self.gwasmapper = GWASMapper() if gwasmapper is None else gwasmapper
        self.adjacencymapper = AdjacencyMapper(mapping_file=config.input.adjacency_mappings) if adjacencymapper is None else adjacencymapper

        self.gwasmappings = self.gwasmapper.get_mappings(tags=phenotype_tag, fields=phenotype_fields)
        self.adjacencies = self.adjacencymapper.get_mappings(tags=adjacency_tag, fields=adjacency_fields)

        self.distinct_phenotypes, self.distinct_mappings = self.get_distinct_mappings()

        self.phenotype_tag = phenotype_tag
        self.adajcency_tag = adjacency_tag

        self.merge = merge

        if not merge and (len(self.distinct_mappings) > 1 or len(self.adjacencies) > 1):
            self.mode = "panorama"
            self.preprocessor = PreProcessor(self.config, [self.distinct_mappings[0]], [self.adjacencies[0]]) if preprocessor is None else preprocessor
        else:
            self.mode = "focus"
            preprocessor = PreProcessor(self.config, self.gwasmappings, self.adjacencies) if preprocessor is None else preprocessor
            self.diagnostic = GraphDiagnostic(preprocessor.get_graph(features=False))

        self.details = ["components", "paths", "degrees", "homophily", "metrics"]

    def get_diagnostics(self, detail: str = "", save=True, **kwargs):
        if detail in ["", "all"] and self.mode == "panorama":
            raise ValueError("if multiple adjacencies and or phenotypes are passed, then only one detail can be plotted. Choose from: {}".format(self.details))

        if self.mode == "panorama":
            fig, ax = self.get_panorama_diagnostics(detail, **kwargs)
        else:
            title = 'Phenotype: {}\nAdjacency: {}'.format(self.distinct_phenotypes[0], self.adjacencies[0]["name"])
            fig, ax = self.diagnostic.get_diagnostics(detail, **kwargs)
            plt.suptitle(title)

        if save:
            if not os.path.exists(self.config.model.plot_dir):
                os.makedirs(self.config.model.plot_dir)
            plt.savefig(os.path.join(self.config.model.plot_dir, "{}_{}_{}.png".format(self.config.name, detail, self.mode)), dpi=300)

        return fig, ax

    def get_distinct_mappings(self):

        distinct_mappings = []
        distinct_phenotypes = []
        for mapping in self.gwasmappings:
            if mapping["phenotype"] not in distinct_phenotypes:
                distinct_phenotypes.append(mapping["phenotype"])
                distinct_mappings.append(mapping)

        return distinct_phenotypes, distinct_mappings

    def get_panorama_diagnostics(self, detail: str, save=True, **kwargs):

        _, distinct_mappings = self.get_distinct_mappings()
        num_phenotypes = len(distinct_mappings)
        num_adjacencies = np.sum([2 if adjacency["directed"] else 1 for adjacency in self.adjacencies])
        fig, axes = plt.subplots(num_phenotypes, num_adjacencies, figsize=(4 * num_adjacencies, (4 * num_phenotypes)), sharex=False, sharey=False)

        adjacencies = []
        for adjacency in self.adjacencies:
            adjacencies.append(adjacency)
            if adjacency["directed"] == "True":
                adjacencies.append(adjacency)
        previoues_adjacency = ""

        nrows = 1 if len(axes.shape) == 1 else axes.shape[0]
        ncols = axes.shape[0] if len(axes.shape) == 1 else axes.shape[1]

        for row_idx, phenotype in zip(range(nrows), distinct_mappings):

            phenotype_tag = phenotype["phenotype"]
            self.preprocessor.assign_new_ground_truth([phenotype], compile=True)

            for col_idx, adjacency in zip(range(ncols), adjacencies):
                adjacency_tag = adjacency["name"]
                self.preprocessor.assign_new_adjacencies([adjacency], compile=True, wipe=True)
                second_time = True if previoues_adjacency == adjacency_tag else False
                previoues_adjacency = adjacency_tag

                if adjacency["directed"] == "True":
                    if second_time:
                        if detail.lower() in ["path", "paths"]:
                            kwargs.update({"symmetrize": True})
                        elif detail.lower() in ["degree", "degrees"]:
                            kwargs.update({"direction": "out"})
                            adjacency_tag += " (out)"
                        previoues_adjacency = ""
                    else:
                        if detail.lower() in ["path", "paths"]:
                            kwargs.update({"symmetrize": False})
                        elif detail.lower() in ["degree", "degrees"]:
                            kwargs.update({"direction": "in"})
                            adjacency_tag += " (in)"

                diagnostic = GraphDiagnostic(self.preprocessor.get_graph(features=False))

                if detail.lower() in ["component", "components"]:
                    detail_func = diagnostic.check_components
                elif detail.lower() in ["path", "paths"]:
                    detail_func = diagnostic.check_paths
                elif detail.lower() in ["degree", "degrees"]:
                    detail_func = diagnostic.check_degrees
                elif detail.lower() == "homophily":
                    detail_func = diagnostic.check_homophily
                elif detail.lower() in ["metric", "metrics"]:
                    detail_func = diagnostic.check_metrics
                else:
                    raise ValueError("No Detail '{}'. Only the following detail views are implemented: {}".format(detail, self.details))

                try:
                    fig, ax = detail_func(fig=fig, ax=axes[row_idx, col_idx], **kwargs)
                except IndexError:
                    fig, ax = detail_func(fig=fig, ax=axes[col_idx], **kwargs)

                ax.set_title('Phenotype: {}\nAdjacency: {}'.format(" ".join([word.capitalize() for word in phenotype_tag.split("_")]), adjacency_tag if not self.merge else self.adajcency_tag))

                plt.tight_layout()

        return fig, ax
