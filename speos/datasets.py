import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
import torch
import os
import torch_geometric.data as pyg_data
from speos.utils.logger import setup_logger
from speos.preprocessing.preprocessor import PreProcessor
import speos.utils.path_utils as pu


class GeneDataset(InMemoryDataset):
    def __init__(self, mappings, adjacencies, name, config, holdout_size: float = 0.5, transform=None, pre_transform=None):
        self.root = config.input.save_dir
        self.save = config.input.save_data
        self.name = name
        self.config = config
        self.holdout_size = holdout_size
        self.preprocessor = PreProcessor(config, mappings, adjacencies)
        self.is_multigraph = False
        super(GeneDataset, self).__init__(self.root, transform, pre_transform)
        if self.save:
            if torch.cuda.is_available():
                self.data = torch.load(self.processed_paths[0])
            else:
                self.data = torch.load(
                    self.processed_paths[0], map_location=torch.device('cpu'))
            self.node_df = pd.read_csv(
                self.processed_paths[1], sep="\t", header=0, index_col=0)
        else:
            try:
                self.data = self._data
            except AttributeError:
                self.process()
                self.data = self._data
            del self._data

        logger = setup_logger(config, __name__)
        logger.info(self.data)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # a list of files in the processed_dir which needs to be found in order to skip the processing
        return pu.processed_data_filename(self.config)

    @property
    def processed_dir(self):
        return pu.processed_data_dir(self.config)

    def process(self):

        X, y, adj = self.preprocessor.get_data()
        df = self.preprocessor.get_node_df(["entrez", "hgnc", "ensembl"])

        indices = np.array(range(len(y)))
        if self.holdout_size > 0:
            train_indices, holdout_indices = train_test_split(
                indices, test_size=self.holdout_size, stratify=y)
        else:
            train_indices = indices

        train_mask = np.array([index in train_indices for index in indices])
        holdout_mask = ~train_mask

        try:
            if self.holdout_size > 0:
                test_indices, val_indices = train_test_split(
                indices[holdout_mask], test_size=0.5, stratify=y[holdout_mask])
            else:
                test_indices, val_indices = [], []
            test_mask = np.array([index in test_indices for index in indices])
            val_mask = np.array([index in val_indices for index in indices])
        except ValueError:
            train_mask = np.ones_like(train_mask).astype(np.bool8)
            val_mask = np.zeros_like(train_mask).astype(np.bool8)
            test_mask = np.zeros_like(train_mask).astype(np.bool8)

        assert np.sum(train_mask + val_mask + test_mask) == len(indices)

        data = self.prepare_data(X, y, adj, train_mask, val_mask, test_mask)

        if self.save:
            torch.save(data, self.processed_paths[0])
            df.to_csv(self.processed_paths[1], sep="\t")
        else:
            self.node_df = df
            self._data = data

    def prepare_data(self, X, y, adj, train_mask, val_mask, test_mask):

        if type(adj) == dict:
            if len(adj.keys()) == 1:
                adj = np.asarray(list(adj.values())).squeeze()
            else:
                logger = setup_logger(self.config, __name__)
                logger.warning("Adjacency data contains {} matrices. Trying to handle them as a single matrix.".format(len(adj.keys())))
                adj = np.concatenate(list(adj.values()), 1).squeeze()

        data = pyg_data.Data(x=torch.tensor(X, dtype=torch.double),
                             edge_index=torch.tensor(adj, dtype=torch.long),
                             y=torch.tensor(y, dtype=torch.double),
                             train_mask=torch.tensor(train_mask),
                             test_mask=torch.tensor(test_mask),
                             val_mask=torch.tensor(val_mask))

        return data


class MultiGeneDataset(GeneDataset):
    def __init__(self, mappings, adjacencies, name, config, holdout_size: float = 0.05, transform=None, pre_transform=None):
        super().__init__(mappings, adjacencies, name, config, holdout_size, transform, pre_transform)
        self.is_multigraph = True

    def prepare_data(self, X, y, adj, train_mask, val_mask, test_mask):

        data = pyg_data.HeteroData(
            {"gene": {"x": torch.tensor(X, dtype=torch.double)},
             "x": torch.tensor(X, dtype=torch.double),
             "y": torch.tensor(y, dtype=torch.double),
             "train_mask": torch.tensor(train_mask),
             "test_mask": torch.tensor(test_mask),
             "val_mask": torch.tensor(val_mask)
             }
        )

        for adj_name in self.preprocessor.adjacency_dict.keys():
            data["gene", adj_name, "gene"].edge_index = torch.tensor(adj[adj_name])

        return data

    def process(self):
        return super().process()


class DatasetBootstrapper:
    def __init__(self, mappings, adjacencies, name, config, holdout_size: float = 0.05):
        if len(adjacencies) > 1 or config.input.force_multigraph:
            self.dataset = MultiGeneDataset(mappings, adjacencies, name, config, holdout_size)
        else:
            self.dataset = GeneDataset(mappings, adjacencies, name, config, holdout_size)

    def get_dataset(self):
        return self.dataset


class MultiData(pyg_data.HeteroData):
    """
        Data object for MultiGraphs, i.e. graphs with only one node type but multiple edge types.
        All attributes except for those provided in exclude_keys are mapped directly to the class itself.
        This way you have easy access to node attributes and dont have to specify the (single) node type every time.
        e.g.: instead of data["node_type"].x you can just write data.x
    """
    def __init__(self, *args, exclude_keys=["edge_index"], **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude_keys = exclude_keys

    def make_compatible(self):
        assert len(self.node_types) <= 1, "MultiData is used only for multiple edge types and one single node type. For multiple node types use HeteroData"
        for key in self.keys:
            if key not in self.exclude_keys and key != "exclude_keys":
                setattr(self, key, self[self.node_types[0]][key])
