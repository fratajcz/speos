from sklearn.preprocessing import robust_scale
from speos.utils.logger import setup_logger
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import extensions.preprocessing as hooks


class PreProcessor:
    def __init__(self,
                 config,
                 mapping_list: list,
                 adjacency_list: list,
                 translation_table: str = "data/hgnc_official_list.tsv",
                 expression_files=["./data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct",
                                   "./data/human_protein_atlas_rna_blood_cell.tsv"],
                 name: str = "RBC",
                 extension_inputs: str = "./extensions/datasets.json"):

        self.name = name
        self.config = config
        self.mapping_list = mapping_list
        self.translation_table_path = translation_table
        self.translation_table = None
        self.G = None
        self.expression_files = expression_files
        self.num_random_features = 100
        self.features_list = []
        self.logger_args = [config, __name__]
        self.pos_idx = []
        self.neg_idx = []
        self.hgnc_key, self.entrez_key, self.ensembl_key = "hgnc", "entrez", "ensembl"
        self.graph_is_built = False
        self.has_features = False
        self.has_labels = False
        self.ensembl2id = None
        self.entrez2id = None
        self.hgnc2id = None
        self.assign_new_adjacencies(adjacency_list, compile=False)
        self.assign_new_ground_truth(mapping_list, compile=False)
        self.read_extension_inputs(extension_inputs)

    def build_graph(self, features=True, use_embeddings=None):

        self.read_translation_table()

        self.build_conversion_dicts()

        if len(self.expression_files) > 0:
            self.build_expression_table()

        self.G = nx.MultiDiGraph()

        node_list = self.get_node_list()

        self.G.add_nodes_from(node_list)

        self.add_adjacencies()

        self.graph_is_built = True

        self.add_y_label()

        if features:
            self.add_x_features(use_embeddings=use_embeddings)

        logger = setup_logger(*self.logger_args)
        logger.info(nx.info(self.G))

    def get_data(self):
        if not self.graph_is_built:
            # if graph has not been set up first
            self.build_graph()

        if not self.has_features:
            # if features arent loaded yet
            self.add_x_features()

        if not self.has_labels:
            # if labels arent loaded yet
            self.add_y_label()

        X, y, adj = self.format_for_pygeo()

        logger = setup_logger(*self.logger_args)
        logger.info("Number of positives in ground truth {}: {}".format(
            self.ground_truth[0], y.sum()))

        return X, y, adj

    def get_node_list(self) -> list:
        return [(i, {self.entrez_key: series[self.entrez_key], self.ensembl_key: series[self.ensembl_key], self.hgnc_key:series[self.hgnc_key], "y": 0, "x": []})
                for i, series in self.translation_table.iterrows()]

    def read_translation_table(self, path=None, hgnc_col="symbol", entrez_col="entrez_id", ensembl_col="ensembl_gene_id", sep="\t") -> None:

        if path is None:
            path = self.translation_table_path

        df = pd.read_csv(path, sep=sep, header=0, usecols=[hgnc_col, entrez_col, ensembl_col])
        df.rename(columns={hgnc_col: self.hgnc_key, entrez_col: self.entrez_key, ensembl_col: self.ensembl_key}, inplace=True)
        self.translation_table = df

    def build_conversion_dicts(self) -> None:

        if self.translation_table is None:
            self.read_translation_table()

        self.ensembl2id = {self.translation_table[self.ensembl_key][i]: i for i in range(
            len(self.translation_table))}
        self.entrez2id = {self.translation_table[self.entrez_key][i]: i for i in range(
            len(self.translation_table))}
        self.hgnc2id = {self.translation_table[self.hgnc_key][i]: i for i in range(
            len(self.translation_table))}

        self.id2hgnc = {value: key for key, value in self.hgnc2id.items()} 

    def contains_directed_graphs(self) -> bool:
        """ Returns True if any of the adjacency matrices is directed """
        return any([_adjacency["directed"] for _adjacency in self.adjacency_list])

    def contains_only_directed_graphs(self) -> bool:
        """ Returns True if all of the adjacency matrices is directed """
        return all([_adjacency["directed"] for _adjacency in self.adjacency_list])

    def reindex(self) -> None:
        del self.ensembl2id
        del self.entrez2id
        del self.hgnc2id
        
        self.G = nx.relabel.convert_node_labels_to_integers(
            self.G, first_label=0, ordering='default')

        self.hgnc2id = {self.G.nodes[i]["hgnc"]: i for i in range(len(self.G.nodes))}
        self.id2hgnc = {i: self.G.nodes[i]["hgnc"] for i in range(len(self.G.nodes))}
        self.ensembl2id = {self.G.nodes[i]["ensembl"]: i for i in range(len(self.G.nodes))}
        self.entrez2id = {self.G.nodes[i]["entrez"]: i for i in range(len(self.G.nodes))}

        self.find_pos_and_neg_idx()

    def assign_new_adjacencies(self, adjacency_list, compile=False, wipe=True) -> None:
        if type(adjacency_list) not in [list, tuple, set]:
            adjacency_list = [adjacency_list]
        self.adjacency_list = adjacency_list
        logger = setup_logger(*self.logger_args)
        logger.info("Using Adjacency matrices: " + str([adjacency["name"] for adjacency in self.adjacency_list]))
        self.adjacency_dict = {}

        if self.G is not None and wipe:
            self.G.remove_edges_from(list(self.G.edges()))

        if compile:
            self.add_adjacencies()

    def add_adjacencies(self) -> None:
        for adjacency_mapping in self.adjacency_list:
            self.adjacency_dict.update({adjacency_mapping["name"]: len(self.adjacency_dict.keys())})

            edge_list = self.handle_ppi(adjacency_mapping)

            self.G.add_edges_from(edge_list)

    def handle_ppi(self, mapping: dict) -> list:
        edge_list = []
        adjacency = pd.read_csv(mapping["file_path"], sep=mapping["sep"], header=0, usecols=[mapping["target"], mapping["source"]])

        if mapping["symbol"] == self.hgnc_key:
            idx_dict = self.hgnc2id
        elif mapping["symbol"] == self.entrez_key:
            idx_dict = self.entrez2id
        elif mapping["symbol"] == self.ensembl_key:
            idx_dict = self.ensembl2id
            if "ENSP" in adjacency[mapping["source"]][0]:
                adjacency = self.translate_protein_to_gene(adjacency)
        else:
            raise ValueError("Currently only handling for entrez, ensembl and hgnc symbols is implemented, you passed symbol of type {}".format(mapping["symbol"]))

        if self.config.input.randomize_adjacency_percent > 0:
            adjacency = self.xswap(adjacency, self.config.input.randomize_adjacency_percent / 100)

        for _, series in adjacency.iterrows():
            try:
                edge_list.append(
                    (idx_dict[series[mapping["source"]]], idx_dict[series[mapping["target"]]], self.adjacency_dict[mapping["name"]]))
                if not mapping["directed"]:
                    # if the adjacency is undirected, also add the inverse edge
                    edge_list.append(
                        (idx_dict[series[mapping["target"]]], idx_dict[series[mapping["source"]]], self.adjacency_dict[mapping["name"]]))
            except KeyError:
                pass
        return edge_list
    
    @classmethod
    def xswap(self, adjacency: pd.DataFrame, randomize_factor: float) -> pd.DataFrame:
        """ Does X Swaps on the adjacency until randomize_factor rows have been changed.
            Assumes that adjacency is a pandas Dataframe with one edge per row and the tail node being the last element of each row.
            If there is an uneven number of edges to reach the threshold, the last edge will remain unswapped """

        from sklearn.model_selection import train_test_split
        assert len(adjacency) > 4
        if randomize_factor == 1:
            rows_to_swap = np.asarray(range(len(adjacency)))
            np.random.shuffle(rows_to_swap)
        else:
            _, rows_to_swap = train_test_split(np.asarray(range(len(adjacency))), test_size=randomize_factor, )
        _adjacency = adjacency.copy(deep=True)
        if len(rows_to_swap) % 2 == 1:
            rows_to_swap = rows_to_swap[:-1]
        swapped_row_indices_mask = [i - 1 if i % 2 == 1 else i + 1 for i in range(len(rows_to_swap))]
        swapped_row_indices = rows_to_swap[swapped_row_indices_mask]
        adjacency_values = _adjacency.values
        adjacency_values[rows_to_swap, 1] = adjacency_values[swapped_row_indices, 1]

        return _adjacency

    def translate_protein_to_gene(self, adjacency: pd.DataFrame):

        table = pd.read_csv("data/protein_gene_table.tsv", header=0, sep="\t")
        protein_to_gene_dict = {row[0]: row[1] for _, row in table.iterrows()}
        adj_dict = adjacency.to_dict("list")
        new_adj_dict = {}
        genes_A = []
        genes_B = []
        for i, row in adjacency.iterrows():
            try:
                genes = [protein_to_gene_dict[protein.split(":")[-1]] for protein in row]
            except KeyError:
                continue
            genes_A.append(genes[0])
            genes_B.append(genes[1])

        new_dict = {key: genes for key, genes in zip(adj_dict.keys(), [genes_A, genes_B])}

        new_adjacency = pd.DataFrame.from_dict(new_dict, orient="columns")

        return new_adjacency

    def build_expression_table(self) -> None:
        self.expression_table = None

        for expression_file in self.expression_files:

            if "gtex" in expression_file.lower():
                new_expression_table = pd.read_csv(
                    expression_file, sep="\t", header=0, index_col=0, skiprows=[0, 1])
                new_expression_table.drop("Description", axis=1, inplace=True)

                if "." in new_expression_table.index[1]:
                    new_expression_table.index = [index.split(
                        ".")[0] for index in new_expression_table.index]

                if self.config.input.log_expression:
                    new_expression_table = np.log(new_expression_table + (new_expression_table[new_expression_table > 0].min() / 2))

                self.join_expression_tables(new_expression_table)

            if "human_protein_atlas" in expression_file.lower():
                raw_table = pd.read_csv(
                    expression_file, sep="\t", header=0, index_col=0)

                genes = list(set(raw_table.index.tolist()))
                tissues = list(set(raw_table["Blood cell"].tolist()))
                expression = np.asarray(raw_table["NX"].tolist()).reshape(
                    (len(genes), len(tissues)))

                new_expression_table = pd.DataFrame(
                    data=expression, index=genes, columns=tissues)
                
                if self.config.input.log_expression:
                    new_expression_table = np.log(new_expression_table + (new_expression_table[new_expression_table > 0].min() / 2))

                self.join_expression_tables(new_expression_table)

    def join_expression_tables(self, new_table) -> None:
        if self.expression_table is None:
            self.expression_table = new_table
        else:
            self.expression_table = self.expression_table.join(
                new_table, how="inner")

    def add_x_features(self, gwas_features=["NSNPS", "ZSTAT", "P"], use_embeddings=None, dummy=False) -> None:
        ''' this removes nodes that do not have all the features that we request.
            Thus, node indices before and after this method call can change and usually do.
        '''

        feature_df_list = [pd.read_csv(mapping["features_file"], sep=" ", header=0)
                           for mapping in self.mapping_list if mapping["features_file"] != ""]

        if len(feature_df_list) > 0:

            columns = feature_df_list[0].columns.tolist()
            column_indices = [columns.index(feature) for feature in gwas_features]

            for feature_df in feature_df_list:
                feature_df.index = feature_df["GENE"].tolist()

        if use_embeddings is None:
            use_embeddings = self.config.input.use_embeddings

        if use_embeddings:
            from gensim.models import KeyedVectors
            wv = KeyedVectors.load(self.config.input.embedding_path, mmap='r')

        if len(self.additional_inputs) > 0:
            addtl_dfs = [getattr(hooks, addtl_input["function"])(*addtl_input["args"], **addtl_input["kwargs"]) for addtl_input in self.additional_inputs]
            addtl_dfs_identifiers = [addtl_input["identifier"].lower() for addtl_input in self.additional_inputs]
            
            for identifier in addtl_dfs_identifiers:
                if identifier not in [self.hgnc_key, self.ensembl_key, self.entrez_key]:
                    raise ValueError("The identifier Key of additional inputs has to be one of {}".format([self.hgnc_key, self.ensembl_key, self.entrez_key]))

        if self.config.input.use_gwas or self.config.input.use_expression or use_embeddings or len(self.additional_inputs) > 0:
            random_run = False
        else:
            # if we neither use gwas, nor expression nor embedding features, use random features instead
            random_run = True
            random_features = np.random.rand(len(list(self.G.nodes)), self.num_random_features)

        if not dummy:
            for i, node in enumerate(list(self.G.nodes(data=True))):
                try:
                    if self.config.input.use_gwas:
                        for feature_df in feature_df_list:
                            data_row = feature_df.loc[int(node[1]["entrez"]), :]
                            for index in column_indices:
                                node[1]["x"].append(np.float64(data_row[index]))
                    if len(self.expression_files) > 0 and self.config.input.use_expression:
                        expression_levels = np.float64(
                            self.expression_table.loc[node[1][self.ensembl_key]].tolist())
                        node[1]["x"].extend(expression_levels)
                    if use_embeddings:
                        try:
                            embedding = wv[node[1]["hgnc"]]
                        except KeyError:
                            embedding = np.zeros_like(wv[wv.index_to_key[0]])
                        node[1]["x"].extend(embedding)
                    if random_run:
                        node[1]["x"].extend(random_features[i, :].tolist())
                    if len(self.additional_inputs) > 0:
                        for df, identifier in zip(addtl_dfs, addtl_dfs_identifiers):
                            node[1]["x"].extend(df.loc[node[1][identifier]].tolist())

                except ValueError:
                    # if there is no entrez id (it is set to nan) of a node, remove it since we can't add features to it
                    self.G.remove_node(node[0])
                except KeyError:
                    # if there is no info about a node in the magma file, remove it
                    self.G.remove_node(node[0])

        if self.config.input.use_gwas:
            for gwas_name in self.gwas_names:
                for gwas_feature in gwas_features:
                    self.features_list.append(" ".join((gwas_feature, gwas_name)))

        if len(self.expression_files) > 0 and self.config.input.use_expression:
            self.features_list.extend(self.expression_table.columns)

        if use_embeddings:
            self.features_list.extend(["EmbDim{}".format(i)
                                      for i in range(len(wv[wv.index_to_key[0]]))])

        if random_run:
            self.features_list.extend("Rand{}".format(i) for i in range(self.num_random_features))

        if not dummy:
            # reindex so indices don't skip numbers after deletion of nodes
            self.reindex()

        self.has_features = True

    def add_y_label(self) -> None:
        if not self.graph_is_built:
            self.build_graph(features=False)

        try:
            known_positives_set = getattr(hooks, self.mapping_list[0]["function"])(*self.mapping_list[0]["args"], **self.mapping_list[0]["kwargs"])
        except KeyError:
            known_positives_df = pd.read_csv(self.mapping_list[0]["ground_truth"], sep="\t", names=[
                                      "chromosome", "start", "end", "symbol", "strand"])
            known_positives_set = set(known_positives_df["symbol"].tolist())

        try:
            if self.mapping_list[0]["symbol"].lower() == "entrez":
                translator = self.entrez2id
            elif self.mapping_list[0]["symbol"].lower() == "ensembl":
                translator = self.ensembl2id
            elif self.mapping_list[0]["symbol"].lower() == "hgnc":
                raise KeyError
            else:
                raise ValueError("We only support translation of extension-labels from entrez and ensembl to hgnc.")
            known_positives_set = set([self.id2hgnc[translator[symbol]] for symbol in known_positives_set])
        except KeyError:
            pass

        self.pos_idx = []
        self.neg_idx = []

        for node in self.G.nodes(data=True):
            if node[1][self.hgnc_key] in known_positives_set:
                node[1]["y"] = 1
                self.pos_idx.append(node[0])
            else:
                node[1]["y"] = 0
                self.neg_idx.append(node[0])

        self.has_labels = True

    def find_pos_and_neg_idx(self):
        self.pos_idx = []
        self.neg_idx = []

        try:
            for node in self.G.nodes(data=True):
                if node[1]["y"] == 1:
                    self.pos_idx.append(node[0])
                else:
                    self.neg_idx.append(node[0])
        except (KeyError, AttributeError):
            self.add_y_label()

        return self.pos_idx, self.neg_idx

    def assign_new_ground_truth(self, mapping_list, compile=True) -> None:
        """
        Goes through mapping list and extracts new ground truth.

        If compile is set to true, calls self.add_y_label and adds labels to nodes in the graph.
        To do this, the graph has to be already built.

        Explicitely call this method only when you want to plot the same adjacency with different labels.
        For ML runs, initialize graphs from scratch each time.
        """

        # quick check that all mappings use the same ground truth
        ground_truth = [mapping["ground_truth"] for mapping in mapping_list]
        if len(set(ground_truth)) > 1:
            raise ValueError("We can use only one ground truth as labels, but you used {}".format(set(ground_truth)))
        elif len(set(ground_truth)) == 0:
            raise ValueError("No ground truth labels found, please check the requested tags")

        self.ground_truth = ground_truth
        self.mapping_list = mapping_list

        logger = setup_logger(*self.logger_args)
        logger.info("Using {} mappings with ground truth {} ".format(
            len(self.mapping_list), self.ground_truth[0]))

        if compile:
            self.add_y_label()

    def format_for_pygeo(self):
        all_edges = np.asarray(self.G.edges)
        adj = {}
        for name, i in self.adjacency_dict.items():
            mask = all_edges[:, 2] == i
            try:
                adj.update({name: np.asarray((all_edges[:, 0][mask], all_edges[:, 1][mask]))})
            except ValueError as e:
                logger = setup_logger(*self.logger_args)
                logger.error(str(e) + ' Skipping adjacency {}'.format(name))
                continue

        X = []
        y = []

        for node in self.G.nodes(data=True):
            X.append(node[1]["x"])
            y.append(node[1]["y"])

        X = np.array(X).astype(np.float64)
        X = robust_scale(X, axis=0)
        y = np.array(y)

        return X, y, adj

    def inspect(self, plot=False):

        x_dict = nx.get_node_attributes(self.G, 'x')
        z_mendel = []
        z_random = []
        z_neighbors_mendel = []
        z_neighbors_random = []

        for node in self.G.nodes(data=True):
            if node[1]["y"] == 1:
                z_mendel.append(node[1]["x"][2])
                neighbors = [n for n in self.G[node[0]]]
                z_neighbors_mendel.extend(
                    [x_dict[neighbor][2] for neighbor in neighbors])

        n_random = len(z_mendel)
        random_indices = np.random.randint(
            0, high=len(self.G.nodes()), size=n_random)

        for node in random_indices:

            z_random.append(x_dict[node][2])
            neighbors = [n for n in self.G[node]]
            z_neighbors_random.extend([x_dict[neighbor][2]
                                      for neighbor in neighbors])

        if plot:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

            sns.violinplot(y=z_neighbors_mendel + z_neighbors_random, x=["mendel"] * len(
                z_neighbors_mendel) + ["random"] * len(z_neighbors_random), cut=0)

            fig.savefig("eda/violin_neighborhood_z_{}.png".format(self.name))
            fig.clf()
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

            sns.violinplot(
                y=z_mendel + z_random, x=["mendel"] * len(z_mendel) + ["random"] * len(z_random), cut=0)

            fig.savefig("eda/violin_z_{}.png".format(self.name))
            fig.clf()

        return z_mendel, z_random, z_neighbors_mendel, z_neighbors_random

    def get_feature_names(self):
        if len(self.features_list) == 0:
            self.build_expression_table()
            self.add_x_features(dummy=True)
        return self.features_list

    @property
    def gwas_names(self):
        return [mapping["name"].split("-")[0] for mapping in self.mapping_list]

    def get_node_df(self, attributes=None):
        if attributes is None:
            # TODO get one node and return a df with all his attributes for all nodes
            raise NotImplementedError

        df = pd.DataFrame(columns=attributes, index=self.G.nodes())

        for attribute in attributes:
            values = nx.get_node_attributes(self.G, attribute).values()
            df[attribute] = values

        return df

    def get_graph(self, features=False):
        if not self.graph_is_built:
            self.build_graph(features=features)

        if features and not self.has_features:
            self.build_graph(features=features)

        return self.G

    def dump_edgelist(self, path, symbol="hgnc"):
        import gzip

        if symbol == "hgnc":
            symbol2id = self.hgnc2id
        elif symbol == "entrez":
            symbol2id = self.entrez2id
        elif symbol == "ensembl":
            symbol2id = self.ensembl2id
        else:
            raise ValueError("Only symbols hgnc, entrez and ensembl are implemented, you requested {}".format(symbol))

        id2symbol = {value: key for key, value in symbol2id.items()}
        buffer_size = 1000
        with gzip.open(path, "wt") as file:
            buffer = []
            for i, (head, tail) in enumerate(self.G.edges(data=False)):
                buffer.append("{}\t{}\n".format(id2symbol[head], id2symbol[tail]))
                if i % buffer_size == 0:
                    file.writelines(buffer)
                    buffer = []
            if len(buffer) > 0:
                file.writelines(buffer)

    def get_metrics(self):
        import igraph as ig

        G = self.G.to_undirected()
        G0 = G.subgraph(max(nx.connected_components(G), key=len))
        G = ig.Graph.from_networkx(G0)
        diameter = G.diameter()
        average_shortest_path_length = np.mean(G.shortest_paths())

        results = {"Diameter": diameter, "Average Shortest Path Length": average_shortest_path_length}

        return results

    def read_extension_inputs(self, path):
        self.additional_inputs = []

        with open(path, "r") as file:
            content = file.read()
            additional_inputs = json.loads(content)

        for addtl_input in additional_inputs:
            logger = setup_logger(self.config, __name__)

            try:
                assert hasattr(hooks, addtl_input["function"])
                self.additional_inputs.append(addtl_input)
            except AssertionError:
                logger.warning("Could not find function {} for additional input {} in extensions/preprocessing.py".format(addtl_input["function"], addtl_input["name"]))    

            logger.info("Using {} additional node data sources: {}".format(len(self.additional_inputs), [input["name"] for input in self.additional_inputs]))

    def get_num_relations(self):
        return len(self.adjacency_list)