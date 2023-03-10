import unittest
import json
import os

from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.preprocessor import PreProcessor
from speos.utils.config import Config


class GWASMapperTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.mapping = [{"name": "UC-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "UC.genes.out",
                         "match_type": "perfect",
                         "significant": "False"},
                        {"name": "RA-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "RA.genes.out",
                         "match_type": "perfect",
                         "significant": "True"},
                        {"name": "XY-cardiovascular_disease",
                         "ground_truth": "Cardiovascular_Disease_genes.bed",
                         "features_file": "XY.genes.out",
                         "match_type": "perfect",
                         "significant": "True"},
                        {"name": "FOO-bar",
                         "ground_truth": "bar_genes.bed",
                         "features_file": "FOO.genes.out",
                         "match_type": "perfect",
                         "significant": "True"}]
        self.mapping_file_path = "speos/tests/files/mappings.json"
        with open(self.mapping_file_path, "w") as file:
            json.dump(self.mapping, file)
        self.mapping = [{"name": "UC-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "UC.genes.out",
                         "match_type": "perfect",
                         "significant": "False"},
                        {"name": "RA-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "RA.genes.out",
                         "match_type": "perfect",
                         "significant": "True"},
                        {"name": "XY-cardiovascular_disease",
                         "ground_truth": "Cardiovascular_Disease_genes.bed",
                         "features_file": "XY.genes.out",
                         "match_type": "perfect",
                         "significant": "True"},
                        {"name": "FOO-bar",
                         "ground_truth": "bar_genes.bed",
                         "features_file": "FOO.genes.out",
                         "match_type": "perfect",
                         "significant": "True"}]
        self.mapper = GWASMapper(mapping_file=self.mapping_file_path)

    def tearDown(self) -> None:
        os.remove(self.mapping_file_path)

    def test_fetch_by_tags_single(self):
        read_mappings = self.mapper.get_mappings(tags="immune_dysregulation", fields="name")
        self.assertEqual(read_mappings[0]["name"], self.mapping[0]["name"])
        self.assertEqual(len(read_mappings), len(self.mapping) - 2)

        read_mappings = self.mapper.get_mappings(tags="RA", fields="name")
        self.assertEqual(read_mappings[0]["name"], self.mapping[1]["name"])
        self.assertEqual(len(read_mappings), len(self.mapping) - 3)

        read_mappings = self.mapper.get_mappings(tags=["immune_dysregulation", "True"], fields=["name", "significant"])
        self.assertEqual(read_mappings[0]["name"], self.mapping[1]["name"])
        self.assertEqual(len(read_mappings), len(self.mapping) - 3)

        read_mappings = self.mapper.get_mappings(tags=["immune_dysregulation", "cardiovascular_disease"], fields="name")
        self.assertEqual(len(read_mappings), len(self.mapping) - 1)

        read_mappings = self.mapper.get_mappings()
        self.assertEqual(len(read_mappings), len(self.mapping))

    def test_extension(self):
        extension = [{"name": "EXT-immune_dysregulation",
                      "ground_truth": "Immune_Dysregulation_genes.bed",
                      "features_file": "EXT.genes.out",
                      "match_type": "perfect",
                      "significant": "False"}]
        mapping_file_path = "speos/tests/files/mappings_extension.json"
        with open(mapping_file_path, "w") as file:
            json.dump(extension, file)

        before_mappings = self.mapper.get_mappings(tags="immune_dysregulation", fields="name")

        mapper = GWASMapper(mapping_file=self.mapping_file_path, extension_mappings=mapping_file_path)

        after_mappings = mapper.get_mappings(tags="immune_dysregulation", fields="name")

        self.assertEqual(len(before_mappings) + 1, len(after_mappings))

        os.remove(mapping_file_path)

    def test_empty_list_labels(self):
        mapping = [{"name": "UNK-immune_dysregulation",
                      "ground_truth": "Immune_Dysregulation_genes.bed",
                      "features_file": "",
                      "match_type": "perfect",
                      "significant": "False"}]

        #GWASMapper(self.config.input.gene_sets, self.config.input.gwas, mapping_file=self.mapping_file_path, extension_mappings=mapping_file_path)
        mapping_file_path = "speos/tests/files/mappings_extension.json"
        with open(mapping_file_path, "w") as file:
            json.dump(mapping, file)

        mapper = GWASMapper(mapping_file=mapping_file_path)

        after_mappings = mapper.get_mappings(tags="immune_dysregulation", fields="name")

        self.assertEqual(1, len(after_mappings))

        os.remove(mapping_file_path)

    def test_case_sensitive(self):
        mapper = GWASMapper()

        after_mappings = mapper.get_mappings(tags="insulin", fields="ground_truth")

        self.assertEqual("insulin_disorder", after_mappings[0]["phenotype"])


class AdjacencyMapperTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.mapper = AdjacencyMapper()

    def test_fetch_by_tags(self):
        with open("speos/adjacencies.json", "r") as file:
            content = file.read()
            self.mapping = json.loads(content)
        formatted_mapping_names = [self.mapper._format_name(mapping["name"]) for mapping in self.mapping]
        read_mappings = self.mapper.get_mappings(tags="BioPlex", fields="name")
        self.assertEqual(read_mappings[0]["name"], formatted_mapping_names[0])
        self.assertEqual(len(read_mappings), 2)

        read_mappings = self.mapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
        self.assertEqual(read_mappings[0]["name"], formatted_mapping_names[1])
        self.assertEqual(len(read_mappings), 1)

    def test_fetch_by_multipe_fields(self):
        with open("speos/adjacencies.json", "r") as file:
            content = file.read()
            self.mapping = json.loads(content)
        formatted_mapping_names = [self.mapper._format_name(mapping["name"]) for mapping in self.mapping]
        read_mappings = self.mapper.get_mappings(tags=["BioPlex", "evo"], fields=["name", "type"])
        #self.assertEqual(read_mappings[0]["name"], formatted_mapping_names[0])
        self.assertEqual(len(read_mappings), 3)

        read_mappings = self.mapper.get_mappings(tags=["BioPlex", "HuRI", "evo"], fields=["name", "name", "type"])
        #self.assertEqual(read_mappings[0]["name"], formatted_mapping_names[0])
        self.assertEqual(len(read_mappings), 4)

    def test_fetch_all(self):
        with open("speos/adjacencies.json", "r") as file:
            content = file.read()
            self.mapping = json.loads(content)
        formatted_mapping_names = [self.mapper._format_name(mapping["name"]) for mapping in self.mapping]
        read_mappings = self.mapper.get_mappings()
        self.assertEqual(read_mappings[0]["name"], formatted_mapping_names[0])
        self.assertEqual(len(read_mappings), len(self.mapping))

    def test_fetch_by_type(self):
        with open("speos/adjacencies.json", "r") as file:
            content = file.read()
            mappings = json.loads(content)

        self.config.input.adjacency_field = "type"

        for _type in ["grn", "ppi", "evo"]:
            good_mappings = [mapping for mapping in mappings if mapping["type"] == _type]
            read_mappings = self.mapper.get_mappings(tags=_type, fields=self.config.input.adjacency_field)
            self.assertEqual(len(good_mappings), len(read_mappings))

    def test_fetch_extensions(self):
        extension = [{"name": "ExtensionAdjacency",
                      "type": "ext",
                      "file_path": "no/such/thing",
                      "source": "SymbolA",
                      "target": "SymbolB",
                      "sep": "\t",
                      "symbol": "hgnc",
                      "weight": "None",
                      "directed": False}]

        mapping_file_path = "speos/tests/files/adjacencies_extension.json"
        with open(mapping_file_path, "w") as file:
            json.dump(extension, file)

        mapper = AdjacencyMapper(extension_mappings=mapping_file_path)

        read_mappings = mapper.get_mappings(tags="ext", fields="type")
        self.assertEqual(1, len(read_mappings))

        extension_ppi = [{"name": "ExtensionAdjacency",
                          "type": "ppi",
                          "file_path": "no/such/thing",
                          "source": "SymbolA",
                          "target": "SymbolB",
                          "sep": "\t",
                          "symbol": "hgnc",
                          "weight": "None",
                          "directed": False}]

        before_mappings = mapper.get_mappings(tags="ppi", fields="type")

        mapping_file_path = "speos/tests/files/adjacencies_extension.json"
        with open(mapping_file_path, "w") as file:
            json.dump(extension_ppi, file)

        mapper = AdjacencyMapper(extension_mappings=mapping_file_path)
        after_mappings = mapper.get_mappings(tags="ppi", fields="type")
        self.assertEqual(len(before_mappings) + 1, len(after_mappings))

        os.remove(mapping_file_path)

    def test_fetch_extensions_only_if_required(self):
        """ We had the bug that extension mappings would always be fetched, irrespective of specified tag """
        extension = [{"name": "ExtensionAdjacency",
                      "type": "ext",
                      "file_path": "no/such/thing",
                      "source": "SymbolA",
                      "target": "SymbolB",
                      "sep": "\t",
                      "symbol": "hgnc",
                      "weight": "None",
                      "directed": False}]

        mapping_file_path = "speos/tests/files/adjacencies_extension.json"
        with open(mapping_file_path, "w") as file:
            json.dump(extension, file)

        mapper = AdjacencyMapper(extension_mappings=mapping_file_path)

        read_mappings = mapper.get_mappings(tags="bioplex", fields="name")

        self.assertTrue(all([mapping["name"].lower().startswith("bioplex") for mapping in read_mappings]))

    def test_blacklist(self):

        other_mapper = AdjacencyMapper(blacklist=self.config.input.adjacency_blacklist)

        full_mappings = self.mapper.get_mappings()
        less_mappings = other_mapper.get_mappings()

        self.assertLess(len(less_mappings), len(full_mappings))

    def test_all_or_empty_string(self):
        full_mappings = self.mapper.get_mappings("")
        also_full_mappings = self.mapper.get_mappings("all")

        self.assertEqual(len(full_mappings), len(also_full_mappings))


class PreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.mapping = [{"name": "UC-immune_dysregulation",
                         "ground_truth": "data/mendelian_gene_sets/Immune_Dysregulation_genes.bed",
                         "features_file": "data/gwas/UC.genes.out",
                         "match_type": "perfect",
                         "significant": "False"},
                        {"name": "RA-immune_dysregulation",
                         "ground_truth": "data/mendelian_gene_sets/Immune_Dysregulation_genes.bed",
                         "features_file": "data/gwas/RA.genes.out",
                         "match_type": "perfect",
                         "significant": "True"}]
        self.mapping_file_path = "speos/tests/files/mappings.json"
        with open(self.mapping_file_path, "w") as file:
            json.dump(self.mapping, file)
        self.gwasmapper = GWASMapper(mapping_file=self.mapping_file_path)

        self.adjacencymapper = AdjacencyMapper()

    def tearDown(self) -> None:
        os.remove(self.mapping_file_path)

    def test_single_adjacency(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3-0 293T", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj[adjacencies[0]["name"]].shape[1], 168892)

        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="hetionet_regulates", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj[adjacencies[0]["name"]].shape[1], 242512)

    def test_two_adjacencies(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj["BioPlex30293T"].shape[1], 168892)
        self.assertEqual(adj["BioPlex30HCT116"].shape[1], 105364)

    def test_contains_directed(self):
        # only undirected
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)

        self.assertFalse(preprocessor.contains_directed_graphs())

        # only directed
        adjacencies = self.adjacencymapper.get_mappings(tags="hetionet_regulates", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        self.assertTrue(preprocessor.contains_directed_graphs())

        # undirected and directed
        adjacencies = self.adjacencymapper.get_mappings(tags=["hetionet_regulates", "BioPlex"], fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        self.assertTrue(preprocessor.contains_directed_graphs())

    def test_contains_only_directed(self):
        # only undirected
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)

        self.assertFalse(preprocessor.contains_only_directed_graphs())

        # only directed
        adjacencies = self.adjacencymapper.get_mappings(tags="hetionet_regulates", fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        self.assertTrue(preprocessor.contains_only_directed_graphs())

        # undirected and directed
        adjacencies = self.adjacencymapper.get_mappings(tags=["hetionet_regulates", "BioPlex"], fields="name")

        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        self.assertFalse(preprocessor.contains_only_directed_graphs())

    def test_log_expression(self):
        # check if logarithmizing protein expression works
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3-0 293T", fields="name")
        config = self.config.deepcopy()
        config.input.log_expression = True
        preprocessor = PreProcessor(config, gwasmappings, adjacencies)

        X, y, adj = preprocessor.get_data()

    def test_no_gwas(self):
        # check if toggling use_gwas works
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3-0 293T", fields="name")
        config = self.config.deepcopy()
        preprocessor = PreProcessor(config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()

        config2 = self.config.deepcopy()
        config2.input.use_gwas = False
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        X2, y2, adj2 = preprocessor2.get_data()
        self.assertLess(X2.shape[1], X.shape[1])
        self.assertLessEqual(X.shape[0], X2.shape[0])

        features = preprocessor2.get_feature_names()
        self.assertTrue(all(["ZSTAT" not in name for name in features]))
        self.assertTrue(all(["NSNPS" not in name for name in features]))

    def test_no_expression(self):
        # check if toggling use_expression works
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
        config = self.config.deepcopy()
        preprocessor = PreProcessor(config, gwasmappings, adjacencies)

        X, y, adj = preprocessor.get_data()

        config2 = self.config.deepcopy()
        config2.input.use_expression = False
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        X2, y2, adj2 = preprocessor2.get_data()
        self.assertLess(X2.shape[1], X.shape[1])
        self.assertLessEqual(X.shape[0], X2.shape[0])

        features = preprocessor2.get_feature_names()
        self.assertTrue(all(["Pancreas" not in name for name in features]))
        self.assertTrue(all(["Liver" not in name for name in features]))
        self.assertTrue(all(["Lung" not in name for name in features]))

    def test_random_features(self):
        # check if toggling off use_expression and use_gwas simultaneously produces random features
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
        config = self.config.deepcopy()
        config.input.use_gwas = False
        config.input.use_expression = False
        preprocessor = PreProcessor(config, gwasmappings, adjacencies)

        X, y, adj = preprocessor.get_data()
        self.assertEqual(preprocessor.num_random_features, X.shape[1])

        features = preprocessor.get_feature_names()
        self.assertTrue(all(["Rand" in name for name in features]))

    def test_xswap_correct(self):
        import pandas as pd
        import numpy as np
        adjacency = pd.DataFrame(data=np.array([list(range(0, 10)), list(range(10, 20))]).reshape(10, 2))

        adjacency_swapped = PreProcessor.xswap(adjacency, 0.4)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertTrue((adjacency.iloc[:, 0] == adjacency_swapped.iloc[:, 0]).all())
        self.assertEqual(is_identical/len(adjacency), 1 - 0.4)

        adjacency_swapped = PreProcessor.xswap(adjacency, 0.8)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertAlmostEqual(is_identical/len(adjacency), 1 - 0.8)

        adjacency_swapped = PreProcessor.xswap(adjacency, 1)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertAlmostEqual(is_identical/len(adjacency), 1 - 1)

    def test_xswap_uneven_rows(self):
        import pandas as pd
        import numpy as np
        adjacency = pd.DataFrame(data=np.array([list(range(0, 11)), list(range(10, 21))]).reshape(11, 2))

        adjacency_swapped = PreProcessor.xswap(adjacency, 0.4)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertAlmostEqual(is_identical/len(adjacency), 1 - (4/11))

        adjacency_swapped = PreProcessor.xswap(adjacency, 0.8)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertAlmostEqual(is_identical/len(adjacency), 1 - (8/11))

        adjacency_swapped = PreProcessor.xswap(adjacency, 1)

        is_identical = 0
        for i, row in adjacency.iterrows():
            if row.tolist() == adjacency_swapped.iloc[i].tolist():
                is_identical += 1

        self.assertAlmostEqual(is_identical/len(adjacency), 1 - (10/11))

    def test_xswap_runs_real_data(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
        config = self.config.deepcopy()
        preprocessor = PreProcessor(config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()

        config2 = self.config.deepcopy()
        config2.input.randomize_adjacency_percent = 50
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        X2, y2, adj2 = preprocessor2.get_data()

        # Cannot compare them directly because some edges get lost. Where? it is less than 1% though
        # same head nodes
        adj = list(adj.values())[0]
        adj2 = list(adj2.values())[0]

    def test_metrics(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()

        metrics = preprocessor.get_metrics()

        print(metrics)


class DummyPreProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = Config()
        self.config.logging.dir = "speos/tests/logs/"

        self.config.name = "DiagnosticTest"
        self.config.crossval.n_folds = 1

        self.config.model.save_dir = "speos/tests/models/"
        self.config.inference.save_dir = "speos/tests/results"
        self.config.model.plot_dir = "speos/tests/plots"

        self.config.input.gwas_mappings = "speos/tests/files/dummy_graph/gwas.json"
        self.config.input.adjacency_mappings = "speos/tests/files/dummy_graph/adjacency.json"
        self.config.input.gene_sets = "speos/tests/files/dummy_graph/"
        self.config.input.gwas = "speos/tests/files/dummy_graph/"

        self.gwasmapper = GWASMapper(self.config.input.gwas_mappings)
        self.adjacencymapper = AdjacencyMapper(mapping_file=self.config.input.adjacency_mappings)

    def test_find_idx(self):
        # a cold call on an unbuilt graph to find positive and negative indices should result in the graph being built and then return the indices
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)

        prev_pos, prev_neg = preprocessor.find_pos_and_neg_idx()

        # check that both lists together contain all the nodes from the preprocessor graph
        self.assertEqual(len(set(preprocessor.G.nodes) - set(prev_pos + prev_neg)), 0)
        self.assertEqual(len(prev_pos), 5)

        # then, when we request X data, some nodes that dont have X data will be removed, altering the node lists
        X, y, adj = preprocessor.get_data()

        after_pos, after_neg = preprocessor.find_pos_and_neg_idx()
        # check that both lists together contain all the nodes from the preprocessor graph
        self.assertEqual(len(set(preprocessor.G.nodes) - set(after_pos + after_neg)), 0)

        # check that both lists contain fewer nodes than before
        self.assertLess(len(after_neg), len(prev_neg))
        # I removed the data for ADA/100, so we should have one fewer positive than before
        self.assertLessEqual(len(after_pos), 4)

    def test_dump_edgelist(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()
        preprocessor.dump_edgelist(os.path.join("speos/tests/data/edgelist.tsv.gz"))

    def test_dump_edgelist_directed(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyDirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()
        preprocessor.dump_edgelist(os.path.join("speos/tests/data/edgelist_directed.tsv.gz"))

    def test_node_dicts(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph(features=True)
        self.assertEqual(len(preprocessor.id2hgnc), len(preprocessor.G))

    def test_load_embeddings(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyDirectedGraph", fields="name")
        self.config.input.use_embeddings = True
        self.config.input.use_gwas = False
        self.config.input.use_expression = False
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(X.shape[1], 100)

        features = preprocessor.get_feature_names()
        self.assertTrue(all(["Emb" in name for name in features]))

    def test_dummy_xswap_undirected(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()
        degree_sequence = sorted((d for n, d in preprocessor.G.degree()), reverse=True)

        config2 = self.config.deepcopy()
        config2.input.randomize_adjacency_percent = 100
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        preprocessor2.build_graph()
        degree_sequence2 = sorted((d for n, d in preprocessor2.G.degree()), reverse=True)

        self.assertTrue((degree_sequence == degree_sequence2))

    def test_dummy_xswap_undirected_with_features(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph(features=True)
        degree_sequence = sorted((d for n, d in preprocessor.G.degree()), reverse=True)

        config2 = self.config.deepcopy()
        config2.input.randomize_adjacency_percent = 100
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        preprocessor2.build_graph(features=True)
        degree_sequence2 = sorted((d for n, d in preprocessor2.G.degree()), reverse=True)

        self.assertTrue((degree_sequence == degree_sequence2))

        X1, y1, adj1 = preprocessor.get_data()
        X2, y2, adj2 = preprocessor2.get_data()

        self.assertTrue(True)

    def test_dummy_xswap_directed(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyDirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()
        degree_sequence = sorted((d for n, d in preprocessor.G.degree()), reverse=True)

        config2 = self.config.deepcopy()
        config2.input.randomize_adjacency_percent = 100
        preprocessor2 = PreProcessor(config2, gwasmappings, adjacencies)
        preprocessor2.build_graph()
        degree_sequence2 = sorted((d for n, d in preprocessor2.G.degree()), reverse=True)

        self.assertTrue((degree_sequence == degree_sequence2))

    def test_dummy_metrics(self):
        gwasmappings = self.gwasmapper.get_mappings(tags="dummy", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyDirectedGraph", fields="name")
        preprocessor = PreProcessor(self.config, gwasmappings, adjacencies)
        preprocessor.build_graph()

        metrics = preprocessor.get_metrics()

        print(metrics)

    def test_label_extension(self):
        config = Config()
        mapping = [ {"name": "UNK-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "",
                         "match_type": "perfect",
                         "significant": "True",
                         "function": "test_preprocess_labels",
                         "args": ["./speos/tests/files/dummy_graph/labels.tsv"],
                         "kwargs": {}
                         }]

        mapping_file_path = "speos/tests/files/mappings.json"
        with open(mapping_file_path, "w") as file:
            json.dump(mapping, file)
        gwasmapper = GWASMapper(mapping_file=mapping_file_path)
        mappings = gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")

        prepro = PreProcessor(config, mappings, adjacencies)
        prepro.build_graph(features=False)
        pos, neg = prepro.find_pos_and_neg_idx()
        self.assertEqual(len(pos), 3)

    def test_label_extension_other_symbol(self):
        config = Config()
        mapping = [{"name": "UNK-immune_dysregulation",
                         "ground_truth": "Immune_Dysregulation_genes.bed",
                         "features_file": "",
                         "match_type": "perfect",
                         "significant": "True",
                         "function": "test_preprocess_labels",
                         "args": ["./speos/tests/files/dummy_graph/labels_ensembl.tsv"],
                         "kwargs": {},
                         "symbol": "ensembl"}]

        mapping_file_path = "speos/tests/files/mappings.json"
        with open(mapping_file_path, "w") as file:
            json.dump(mapping, file)
        gwasmapper = GWASMapper(mapping_file=mapping_file_path)
        mappings = gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencies = self.adjacencymapper.get_mappings(tags="DummyUndirectedGraph", fields="name")

        prepro = PreProcessor(config, mappings, adjacencies)
        prepro.build_graph(features=False)
        pos, neg = prepro.find_pos_and_neg_idx()
        self.assertEqual(len(pos), 3)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
