import h5py
import os
import pandas as pd
import numpy as np
import warnings


class DataHandler:
    def __init__(self, file_path, index: list = None, read_only=False):
        self.file_path = file_path
        self.read_only = read_only
        if not read_only:
            self.create_file(index)

    def create_file(self, index: list):
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))
        with h5py.File(self.file_path, "w") as f:
            f.create_dataset("index", (len(index), 1), 'S10', index)

    def get_file_path(self):
        return self.file_path


class ResultsHandler(DataHandler):
    """Stores results as a 3D Cube: foldnr x gene x prediction/explanation.
       Shape is the tuple of dimensions of the dataset.data.x so we know how much space to reserve for faster indexing later"""
    def __init__(self, file_path, n_folds: int = None, shape: tuple = None, explanation: bool = False, index: list = None, read_only=False):
        super(ResultsHandler, self).__init__(file_path, index, read_only)
        if read_only:
            # this will stay open until this instance of ResultsHandler is killed!
            self.file = h5py.File(self.file_path, 'r')
            self.results_dset = self.file["results"]

            try:
                self.explanation_dset = self.file["explanation"]
            except KeyError:
                pass

            self.folds = self.file["folds"]
        else:
            self.n_folds = n_folds
            self.explanation = explanation
            self.explanations_shape = (self.n_folds, shape[0], shape[1]) if explanation else None
            self.results_shape = (self.n_folds, shape[0], 6)
            self.current_results_fold = 0
            self.current_explanation_fold = 0
            with h5py.File(self.file_path, 'a') as file:
                if "results" not in file.keys():
                    results_dset = file.create_dataset("results", data=np.empty(self.results_shape), shape=self.results_shape)
                    # empty container to store the folds names so they can be mapped to indices
                    results_dset.attrs["folds"] = np.empty((self.n_folds,), dtype="S")
                if "explanation" not in file.keys():
                    file.create_dataset("explanation", data=np.empty(self.explanations_shape), shape=self.explanations_shape, dtype=np.float32) if explanation else None
                if "folds" not in file.keys():
                    file.create_dataset("folds", (self.n_folds, 1), 'S10', np.empty((self.n_folds, 1), dtype='S10'))

    def add_results(self, input: pd.DataFrame, name):
        self._add("results", input, name)

    def add_explanations(self, input: pd.DataFrame, name):
        self._add("explanation", input, name)

    def _add(self, dset="results", input: pd.DataFrame = None, name=None):
        assert not self.read_only, "Data cannot be added in read only mode!"
        with h5py.File(self.file_path, 'a') as file:
            values = input.to_numpy(copy=True)
            columns = input.columns.tolist()
            dataset = file.get(dset)
            current_fold = getattr(self, "current_{}_fold".format(dset))
            dataset[current_fold, :, :] = values
            if "{}_columns".format(dset) not in file.keys():
                file.create_dataset("{}_columns".format(dset), data=columns, shape=(len(columns),), dtype="S10")
            folds = file.get("folds")
            if name is None:
                name = str(current_fold)
            folds[current_fold] = name

        if dset == "results":
            self.current_results_fold += 1
        elif dset == "explanation":
            self.current_explanation_fold += 1

    def get_explanation_for_gene(self, hgnc):
        return self._get("explanation", gene=hgnc)

    def get_results_for_gene(self, hgnc):
        return self._get("results", gene=hgnc)

    def get_results_for_fold(self, fold_name_or_int):
        return self._get("results", fold=fold_name_or_int)

    def _get(self, dset="results", fold=None, gene=None):
        try:
            if self.read_only:
                dataset = getattr(self, "{}_dset".format(dset))
                if fold is None:
                    fold_idx = slice(dataset.shape[0])
                else:
                    fold_idx = fold
                if gene is None:
                    gene_idx = slice(dataset.shape[1])
                else:
                    gene_idx = np.where(self.gene_names == gene)[0]

                query = dataset[fold_idx, gene_idx, :]
            else:
                with h5py.File(self.file_path, 'a') as file:
                    dataset = file.get(dset)
                    if fold is None:
                        fold_idx = slice(dataset.shape[0])
                    else:
                        fold_idx = fold
                    if gene is None:
                        gene_idx = slice(dataset.shape[1])
                    else:
                        gene_idx = np.where(file.get("index")[:].astype(str).squeeze() == gene)[0]

                    query = dataset[fold_idx, gene_idx, :]

        except KeyError:
            warnings.warn("{} accessed in Results File but the Results File does not contain any {}. Returning empty List.".format(dset, dset))
            return np.array([])

        return query

    def get_results(self):
        return self._get(dset="results")

    def get_explanation_for_fold(self, fold_name_or_int):
        # retrieving fold by name is buggy
        return self._get("explanation", fold=fold_name_or_int)

    def get_all_results(self, truth_positives=None, predicted_positives=None, train=True, val=False, test=False):
        mask = self.get_all(truth_positives, predicted_positives, train, val, test)
        predicted_index = np.where(self.file["results_columns"][:].astype(str).squeeze() == "probability")[0]
        results = self.results_dset[:, :, predicted_index][mask]
        return results

    def get_all_explanations(self, truth_positives=None, predicted_positives=None, train=True, val=False, test=False):
        mask = self.get_all(truth_positives, predicted_positives, train, val, test)

        return self.explanation_dset[mask]

    @property
    def gene_names(self):
        return self.file.get("index")[:].astype(str).squeeze()

    def close(self):
        self.file.close()
