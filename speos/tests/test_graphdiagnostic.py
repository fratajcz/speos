import unittest
from speos.preprocessing.mappers import GWASMapper
from speos.utils.config import Config
from speos.visualization.diagnosticwrapper import GraphDiagnosticWrapper
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import TestSetup

class PanoramaTest(TestSetup):
    """ This class tests the panorama diagnostics on fake data (i.e without relying on preprocessor)"""

    def setUp(self) -> None:
        super().setUp()
        self.config.name = "DiagnosticTest"

        self.diagnostic = GraphDiagnosticWrapper(config=self.config, phenotype_tag="immune_dysregulation", adjacency_tag=["DummyUndirectedGraph", "DummyUndirectedOtherGraph"])

    def test_panorama_no_detail_fails(self):
        self.assertRaises(ValueError, self.diagnostic.get_diagnostics)

    def test_panorama_wrong_detail_fails(self):
        self.assertRaises(ValueError, self.diagnostic.get_diagnostics, "foo")

    def test_panorama_check_paths(self):
        ppm = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        fig, ax = self.diagnostic.get_diagnostics("paths", save=False, ppm=ppm)
        ax.text(-3, 1, "This should show a bar in neighborhood 1 \nthat is 2 high and contains color for 1", color="red")
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_paths_1.png"))

        ppm = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.uint8)
        fig, ax = self.diagnostic.get_diagnostics("paths", save=False, ppm=ppm)
        ax.text(-3, 1, "This should show a bar in neighborhood 1 \nthat is 3 high and contains color for 2x1 and 1x2\n \
                        and a bar in neighborhood 2 that is \n2 high with colors for 1", color="red")
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_paths_2.png"))

    def test_panorama_check_components(self):
        components, isolates = {0: [[1, 2, 3, 4, 5, 6, 7], [2, 3]], 1: [[8], []]}, [[9, 10, 11], [9]]
        fig, ax = self.diagnostic.get_diagnostics("components", save=False, components=components, isolates=isolates)
        ax.text(-3, 1, "Component 0: 7, 2 of which pos. tiny: 2, 0. disconnected: 3, 1", color="red")
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_components.png"))

    def test_panorama_check_degrees(self):
        degrees = {"Unlabeled": [0, 0, 1, 2], "Positive": [0, 3, 4]}
        fig, ax = self.diagnostic.get_diagnostics("degrees", save=False, degrees=degrees)
        ax.text(1, 10, "Positive: 0,3,4, Unlabeled: 0,0,1,2", color="red")
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_degrees.png"))

    def test_panorama_check_homophily(self):
        confusion = np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float64)
        fig, ax = self.diagnostic.get_diagnostics("homophily", save=False, confusion=confusion)
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_homophily.png"))

    def test_panorama_check_metrics(self):
        metrics = {"Average Positive Distance": (1.0, 0.5),
                   "Average Overall Distance": (1.0,)}
        fig, ax = self.diagnostic.get_diagnostics("metrics", save=False, metrics=metrics)
        fig.set_size_inches(10, 5, forward=True)
        plt.savefig(os.path.join("speos/tests/plots/", "check_metrics.png"))


class FocusTest(TestSetup):
    """ This class tests the focus diagnostics on fake data (i.e without relying on preprocessor)"""

    def setUp(self) -> None:
        super().setUp()
        self.config.name = "DiagnosticTest"

        self.diagnostic = GraphDiagnosticWrapper(config=self.config, phenotype_tag="immune_dysregulation", adjacency_tag="DummyUndirectedGraph")

    def test_focus_wrong_detail_fails(self):
        self.assertRaises(ValueError, self.diagnostic.get_diagnostics, "foo")

    def test_focus_all(self):
        fig, ax = self.diagnostic.get_diagnostics("", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "focus_all.png"))


class DummyGraphTest(TestSetup):
    """ This class employs a small dummy graph and lets the preprocessor build up from scratch to test the interplay between prepro and diagnostic"""

    def setUp(self) -> None:
        super().setUp()

        self.config.name = "DiagnosticTest"

        self.gwasmapper = GWASMapper(mapping_file=self.config.input.gwas)

    def test_dummy_paths_directed(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyDirectedGraph")

        fig, ax = diagnostic.get_diagnostics("paths", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_paths.png"))

    def test_dummy_paths_undirected(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyUndirectedGraph")

        fig, ax = diagnostic.get_diagnostics("paths", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_paths_undirected.png"))

    def test_dummy_degrees_directed(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyDirectedGraph")

        fig, ax = diagnostic.get_diagnostics("degrees", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_degrees.png"))

        fig, ax = diagnostic.get_diagnostics("degrees", save=False, kwargs={"degrees": dict(density=False)})
        ax.text(1, 2, "In: Positive: 0: 2, 1: 3, Out: 0: 4, 2: 1", color="red")
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_degrees_absolute.png"))

    def test_dummy_degrees_undirected(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyUndirectedGraph")

        fig, ax = diagnostic.get_diagnostics("degrees", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_degrees_undirected.png"))

        fig, ax = diagnostic.get_diagnostics("degrees", save=False, kwargs={"degrees": dict(density=False)})
        ax.text(1, 2, "Positive: 0: 1, 1:3, 2:1", color="red")
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_degrees_undirected_absolute.png"))

    def test_dummy_homophily_directed(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyDirectedGraph")

        fig, ax = diagnostic.get_diagnostics("homophily", save=False)
        fig.tight_layout()
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_homophily.png"))

    def test_dummy_homophily_undirected(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyUndirectedGraph")

        fig, ax = diagnostic.get_diagnostics("homophily", save=False)
        plt.savefig(os.path.join("speos/tests/plots/", "dummy_homophily_undirected.png"))

    def test_dummy_components_directed(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyDirectedGraph")

        fig, ax = diagnostic.get_diagnostics("components", save=False)
        plt.savefig(os.path.join(self.config.model.plot_dir, "dummy_components.png"))

    def test_dummy_components_undirected(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyUndirectedGraph")

        fig, ax = diagnostic.get_diagnostics("components", save=False)
        plt.savefig(os.path.join(self.config.model.plot_dir, "dummy_components_undirected.png"))

    def test_dummy_metrics(self):
        diagnostic = GraphDiagnosticWrapper(gwasmapper=self.gwasmapper, config=self.config, phenotype_tag="dummy", adjacency_tag="DummyUndirectedGraph")

        fig, ax = diagnostic.get_diagnostics("metrics", save=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.model.plot_dir, "dummy_metrics_undirected.png"))


if __name__ == '__main__':
    unittest.main(warnings='ignore')
