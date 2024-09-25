import unittest
from speos.utils.config import Config
import shutil
import pathlib

class TestSetup(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree(self.purge_dir, ignore_errors=True)

    def setUp(self):

        self.config = Config() 

        self.config.input.adjacency = "DummyUndirectedGraph"
        self.config.input.gwas_mappings = "speos/tests/files/dummy_graph/gwas.json"
        self.config.input.adjacency_mappings = "speos/tests/files/dummy_graph/adjacency.json"
        self.config.input.gene_sets = "speos/tests/files/dummy_graph/"
        self.config.input.gwas = "speos/tests/files/dummy_graph/gwas.json"

        self.config.logging.level = 10

        self.purge_dir = "speos/tests/temp_" + str(hash(self)) + "/"
        self.config.logging.dir = self.purge_dir + "logs/"
        self.config.pp.save_dir = self.purge_dir + "results"
        self.config.pp.plot_dir = self.purge_dir + "plots"
        self.config.model.save_dir = self.purge_dir + "models/"
        self.config.inference.save_dir = self.purge_dir + "results/"
        self.config.input.save_dir = self.purge_dir + "data/"

        self.config.model.pre_mp.n_layers = 0
        self.config.model.pre_mp.dim = 5
        self.config.model.mp.n_layers = 1
        self.config.model.mp.dim = 5
        self.config.model.post_mp.n_layers = 0
        self.config.model.post_mp.dim = 5

        

        self.setup_dirs = [self.purge_dir + "logs/", self.purge_dir + "results/", self.purge_dir + "plots", self.purge_dir + "models/", self.purge_dir + "data/"]

        

        translation_table_path = "speos/tests/files/dummy_graph/dummy_translation_table.tsv"
        expression_file_paths = ["speos/tests/files/dummy_graph/dummy_gtex_file.tsv", "speos/tests/files/dummy_graph/dummy_human_protein_atlas_file.tsv"]

        self.prepro_kwargs = {"translation_table": translation_table_path,
                              "expression_files": expression_file_paths}

        for dir in self.setup_dirs:
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True) 

        super().setUp()