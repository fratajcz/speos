from speos.experiment import Experiment, InferenceEngine
from speos.utils.logger import setup_logger
import speos.utils.path_utils as pu
import numpy as np
import torch
from copy import deepcopy
import os
import gc


class BaseWrapper():
    def __init__(self, config):
        self.name = config.name
        self.config = config
        self.cleanup_list = []

    def cleanup(self):
        # The wrappers save intermediate data files to distribute them to the runs. This deletes them.
        for item in self.cleanup_list:
            if os.path.exists(item):
                os.remove(item)


class BaggingWrapper(BaseWrapper):
    def __init__(self, config, logger=None):
        super(BaggingWrapper, self).__init__(config)
        self.n_folds = config.crossval.n_folds
        self.base_experiment = Experiment(config, id="bootstrap")
        self.data = self.base_experiment.data

        if config.crossval.seed is not None:
            self.rng = np.random.default_rng(config.crossval.seed)
        else:
            self.rng = np.random.default_rng()
        self.num_individuals = self.data.y.shape[0]
        self.sample_factor = 1.5

    def run(self):
        logger = setup_logger(self.config, __name__ + " (bagging)")
        logger.info(
            "Running Bagging with {} splits.".format(self.n_folds))

        for i in range(self.n_folds):
            self.run_epoch(i)

        if not self.config.input.save_data:
            self.cleanup()

    def run_epoch(self, i):
        logger = setup_logger(self.config, __name__ + " (bagging)")
        logger.info("Start Epoch {}: CUDA Memory allocated: {}; reserved: {}".format(
            i, torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        data = deepcopy(self.data)
        logger.debug("After data copying: CUDA Memory allocated: {}; reserved: {}".format(
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        data.train_mask, data.val_mask = self.resample()

        sub_config = self.config.deepcopy()
        sub_config.name = self.name + self.config.crossval.suffix.format(i)

        sub_config.logging.file = self.name
        path = pu.processed_data_dir(self.base_experiment.config)
        name_pt, name_tsv = pu.processed_data_filename(sub_config)
        torch.save(data, os.path.join(path, name_pt))
        self.cleanup_list.append(os.path.join(path, name_pt))

        df = self.base_experiment.dataset.node_df
        df.to_csv(os.path.join(path, name_tsv), sep="\t")
        self.cleanup_list.append(os.path.join(path, name_tsv))

        split_experiment = Experiment(
            sub_config, resultshandler=self.base_experiment.resultshandler, id=i)
        logger.debug("After experiment creation: CUDA Memory allocated: {}; reserved: {}".format(
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))

        split_experiment.run()

        if self.config.inference.switch:
            split_inference_engine = InferenceEngine(
                sub_config, resultshandler=self.base_experiment.resultshandler)
            split_inference_engine.infer()
            del split_inference_engine

        logger.debug("End Epoch {}: CUDA Memory allocated: {}; reserved: {}".format(
            i, torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        logger.debug("Deleting References.")
        del data
        torch.cuda.empty_cache()
        logger.debug("After data deletion: CUDA Memory allocated: {}; reserved: {}".format(
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        del split_experiment.data
        del split_experiment.model
        del split_experiment
        torch.cuda.empty_cache()
        logger.debug("After experiment deletion: CUDA Memory allocated: {}; reserved: {}".format(
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("After gc call: CUDA Memory allocated: {}; reserved: {}".format(
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))

    def resample(self):

        size = int(self.num_individuals * self.sample_factor)
        drawn = set((self.rng.random((size,)) * self.num_individuals).astype(np.uint16))
        not_drawn = set(range(self.num_individuals)) - drawn
        # initialize as 0
        train = torch.zeros_like(self.data.train_mask)
        val = torch.zeros_like(self.data.val_mask)
        # set to 1 according to random draws
        train[torch.LongTensor(list(drawn))] = 1
        val[torch.LongTensor(list(not_drawn))] = 1
        # set overlaps with test set to 0 again
        train[self.data.test_mask] = 0
        val[self.data.test_mask] = 0

        return train, val


class CVWrapper(BaseWrapper):
    def __init__(self, config, slave=False):
        super(CVWrapper, self).__init__(config)
        self.name = config.name
        self.config = config
        self.n_folds = config.crossval.n_folds

        if slave:
            config = config.deepcopy()
            config.crossval.n_folds = config.crossval.n_folds * (config.crossval.n_folds + 1)

        self.base_experiment = Experiment(config, id="bootstrap")
        self.data = self.base_experiment.data

        self.test_split = self.n_folds if self.config.crossval.hold_out_test else None

        if self.config.crossval.positive_only:
            self.indices = self.split_pos_indices(self.n_folds, self.config.crossval.seed)
        else:
            self.indices = self.split_all_indices(self.n_folds, self.config.crossval.seed)
        

    def run(self):
        logger = setup_logger(self.config, __name__ + " (cv)")
        logger.info(
            "Running Inner Cross Validation with {} splits.".format(self.n_folds))

        for i in range(len(self.indices)):

            if self.indices is not None and i == self.test_split:
                continue

            logger.info("Starting Inner Split {}".format(i))

            sub_config = self.config.deepcopy()
            sub_config.name = self.name + self.config.crossval.suffix.format(i)

            sub_config.logging.file = self.name

            if self.config.training.switch:

                val_mask = self.get_val_mask(i)
                train_mask = self.get_train_mask(i)

                data = deepcopy(self.data)

                data.val_mask = torch.from_numpy(val_mask)
                data.train_mask = torch.from_numpy(train_mask)
                data.test_mask = torch.from_numpy(self.test_mask)
                path = pu.processed_data_dir(self.base_experiment.config)
                name_pt, name_tsv = pu.processed_data_filename(sub_config)
                torch.save(data, os.path.join(path, name_pt))
                self.cleanup_list.append(os.path.join(path, name_pt))

                df = self.base_experiment.dataset.node_df.copy(deep=True)
                df["truth"] = data.y.detach().cpu().numpy().astype(np.uint8)
                df["train"] = train_mask.astype(np.uint8)
                df["val"] = val_mask.astype(np.uint8)
                df["test"] = self.test_mask.astype(np.uint8)
                df.to_csv(os.path.join(
                    path, name_tsv), sep="\t")
                
                self.cleanup_list.append(os.path.join(path, name_tsv))

                split_experiment = Experiment(
                    sub_config, resultshandler=self.base_experiment.resultshandler, id=i)

                split_experiment.run()

                del data
                del split_experiment.model
                del split_experiment.data
                del split_experiment
                gc.collect()
                torch.cuda.empty_cache()

            if self.config.inference.switch:
                split_inference_engine = InferenceEngine(
                    sub_config, resultshandler=self.base_experiment.resultshandler)
                split_inference_engine.infer()
                del split_inference_engine.model
                del split_inference_engine.data
                del split_inference_engine
                gc.collect()
                torch.cuda.empty_cache()

        if not self.config.input.save_data:
            self.cleanup()

        return self.base_experiment.resultshandler.get_file_path()

    def split_indices(self, indices, n_folds, seed):
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        # generate n+1 splits if we want to hold out a test set
        if self.test_split is not None:
            n_folds += 1
        return np.array_split(indices, n_folds)

    def split_pos_indices(self, n_folds, seed):
        pos_indices = np.nonzero(self.data.y.detach().cpu().numpy())[0]
        return self.split_indices(pos_indices, n_folds, seed)

    def split_neg_indices(self, n_folds, seed):
        neg_indices = np.nonzero(1 - self.data.y.detach().cpu().numpy())[0]
        return self.split_indices(neg_indices, n_folds, seed)

    def split_all_indices(self, n_folds, seed):
        pos_indices = self.split_pos_indices(n_folds, seed)
        neg_indices = self.split_neg_indices(n_folds, seed)

        all_indices = []
        for pos_ind, neg_ind in zip(pos_indices, neg_indices):
            all_indices.append(np.concatenate((pos_ind, neg_ind)))
        return all_indices

    def get_val_mask(self, index):
        val_indices = self.indices[index]
        val_mask = np.zeros((self.data.y.shape[0])).astype(np.bool8)
        val_mask[val_indices] = True

        return val_mask

    def get_train_mask(self, index):
        """Anything that is neither Val nor Test is Train"""
        val_mask = self.get_val_mask(index)
        return ~(val_mask + self.test_mask)

    @property
    def test_indices(self):
        return self.indices[self.test_split]

    @property
    def test_mask(self) -> np.ndarray:
        test_mask = np.zeros((self.data.y.shape[0])).astype(np.bool8)
        if self.test_split is not None:
            test_mask[self.test_indices] = True
        return test_mask


class OuterCVWrapper:
    def __init__(self, config):
        self.name = config.name
        self.config = config
        self.n_folds = config.crossval.n_folds + 1

        self.inner_cv = CVWrapper(config, slave=True)

    def run(self):
        logger = setup_logger(self.config, __name__ + " (ocv)")
        logger.info(
            "Running Outer Cross Validation with {} splits.".format(self.n_folds))
        for i in range(self.n_folds):
            logger.info("Starting Outer Split {}".format(i))
            self.inner_cv.name = self.name + \
                self.config.crossval.outer_suffix.format(i)
            self.inner_cv.test_split = i
            self.inner_cv.run()
