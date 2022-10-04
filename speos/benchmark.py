from speos.wrappers import CVWrapper
from speos.utils.metrics import MetricsHelper
from speos.pipeline import Pipeline
import os
import gc
import torch


class TestBench(Pipeline):
    def __init__(self, parameter_file, config_path="", repeats=4):
        super(TestBench, self).__init__(config_path)
        # read parameters that should be benchmarked
        self.read_parameter_list(parameter_file)
        self.resultshandlers = []
        self.configs = []
        self.repeats = repeats

    def run(self):
        self.fit()
        self.compare()

    def fit(self):
        # iterate over parameters and run the same experiment with each of the settings
        for parameters in self.parameter_list:
            for repeat in range(self.repeats):
                # get new config reflecting the parameters
                config = self.adapt_config(parameters)
                # make sure we have a unique name for this run
                config.name = self.config.name + "_" + self.name + "_" + config.name + "rep{}".format(repeat)
                # feed config into CV Pipeline
                cv = CVWrapper(config)
                # run pipeline and gather the results
                try:
                    handler = cv.run()
                except RuntimeError:
                    handler = None
                self.resultshandlers.append(handler)
                self.configs.append(config)

                del cv.base_experiment
                del cv.data
                del cv
                gc.collect()
                torch.cuda.empty_cache()

    def add_resultshandlers(self, handlers: list):
        self.resultshandlers += handlers

    def add_configs(self, configs: list):
        self.configs += configs

    def compile_resultshandlers(self):
        """These are usually added druing fit(), but if the models are already fit, just run this function to recreate resultshandlers paths"""
        if len(self.configs) < len(self.parameter_list):
            self.compile_configs()

        self.resultshandlers = []
        for config in self.configs:
            self.resultshandlers.append(os.path.join(config.inference.save_dir, config.name + ".h5"))

    def compile_configs(self):
        self.configs = []
        for parameters in self.parameter_list:
            for repeat in range(self.repeats):
                # get new config reflecting the parameters
                config = self.adapt_config(parameters)
                # make sure we have a unique name for this run
                config.name = self.config.name + "_" + self.name + "_" + config.name + "rep{}".format(repeat)
                self.configs.append(config)

    def eval(self, **kwargs):
        """Convenience Function that just takes the TestBench's parameter file, recreates the configs and tries to find the resultshandlers of already trained runs.
            Use this function instead of run() if the models are already trained and the results are already there, just the comparison is missing (i.e. due to crashes)"""

        self.compile_configs()
        self.compile_resultshandlers()
        return self.compare(**kwargs)

    def compare(self, save=True, save_path="", additional_masks=None, **kwargs):
        import pandas as pd
        import numpy as np
        from speos.utils.datahandlers import ResultsHandler
        # finally, take all results and draft a comparison
        indices = [[config.name + self.config.crossval.suffix.format(fold) for fold in range(self.config.crossval.n_folds)] for config in self.configs]
        indices = [i for j in indices for i in j]
        try:
            df = pd.DataFrame(data=np.empty((len(self.parameter_list) * self.config.crossval.n_folds * self.repeats, len(self.metrics))),
                            columns=self.metrics,
                            index=indices)
        except ValueError:
            df = pd.DataFrame(data=np.empty((len(self.parameter_list) * self.config.crossval.n_folds * self.repeats, len(self.metrics))),
                            columns=self.metrics)
        comparison = RunComparison(metrics=self.metrics)

        if additional_masks is None:
            additional_masks = [None] * len(self.configs)

        for config, handler_path, additional_mask in zip(self.configs, self.resultshandlers, additional_masks):
            if handler_path is None:
                continue
            handler = ResultsHandler(handler_path, read_only=True)
            values = comparison.eval(handler.get_results(), additional_mask=additional_mask, **kwargs)
            indices = [config.name + self.config.crossval.suffix.format(fold) for fold in range(self.config.crossval.n_folds)]
            for i, index in enumerate(indices):
                df.loc[index] = values[i, :]

        if save_path == "":
            save_path = self.config.name + "_" + self.name + ".tsv"

        if save:
            df.to_csv(save_path, sep="\t", header=True, index=True)

        return df

    def adapt_config(self, parameters):
        config = self.config.deepcopy()
        config.recursive_update(config, parameters)

        return config

    def read_parameter_list(self, parameter_file):
        import yaml
        with open(parameter_file, "r") as file:
            parameter_list = yaml.load(file, Loader=yaml.SafeLoader)

        self.metrics = parameter_list["metrics"]
        self.parameter_list = parameter_list["parameters"]
        self.name = parameter_list["name"]


class RunComparison:

    def __init__(self, metrics: list = ["mrr_filtered", "mean_rank_filtered", "auroc"]):
        self.metrics = metrics

    def eval(self, results_, target="val", additional_mask=None, plot=False, remove_nodes=[]):
        """ Remove nodes is an iterable that contains the indices of nodes to be removed from the results prior to anything else
            Additional mask is a bool array or None. If it is a bool array it will also be masked out during evaluation of metrics
            Note: Additional mask will be coerced with regular mask using AND operation, so it should have 1s where you want to keep nodes and will only
                  produce a subset of the original nodes (i.e. sum(reg_mask) > sum(reg_mask AND add_mask)
                  If additional mask is one-dimensional it will be applied to all folds simultaneously """
        import numpy as np

        #np.delete(results_, remove_nodes, axis=1)

        metrics_array = np.empty((results_.shape[0], len(self.metrics)))

        if target == "val+test":
            masks = np.sum(results_[:, :, 4:6], axis=0).astype(np.bool8)
        elif target == "val":
            masks = results_[:, :, 4].astype(np.bool8)
        elif target == "test":
            masks = results_[:, :, 5].astype(np.bool8)
        elif target == "train":
            masks = results_[:, :, 3].astype(np.bool8)

        if additional_mask is not None:
            masks = masks & additional_mask

        truth = results_[0, :, 0]

        for fold_id in range(results_.shape[0]):
            predictions = results_[fold_id, :, 2]
            metrics = MetricsHelper(truth, masks={target: masks[fold_id, :]}, pred_cutoff=0.7)
            metrics.update(np.asarray(predictions), target)
            try:
                metrics_values = metrics.get_metrics(*self.metrics)
            except (IndexError, ValueError):
                metrics_values = np.empty((len(self.metrics),))
                metrics_values.fill(np.nan)
            metrics_array[fold_id, :] = metrics_values

        return metrics_array

    def eval_bagging(self, results_, p=1, target="val", aggr_func="median"):
        import numpy as np
        if p < 1:
            indices = results_.shape[0]
            import random
            choices = random.choices(range(indices), k=int(indices * p))
            results_ = results_[choices, :, :]
        truth = results_[0, :, 0]
        if target == "val+test":
            mask = np.sum(results_[:, :, 4:6], axis=0)
            predictions = results_[:, :, 2][((results_[:, :, 4] == 1) + (results_[:, :, 5] == 1))]
        elif target == "val":
            mask = np.sum(results_[:, :, 4], axis=0)
            predictions = results_[:, :, 2][(results_[:, :, 4] == 1)]
        elif target == "test":
            mask = np.sum(results_[:, :, 5], axis=0)
            predictions = results_[:, :, 2][(results_[:, :, 5] == 1)]
        elif target == "train":
            mask = np.sum(results_[:, :, 3], axis=0)
            predictions = results_[:, :, 2][results_[:, :, 3] == 1]

        indices = np.where(mask)[0]
        both = np.stack((indices, predictions), axis=1)
        both_sorted = both[both.T[0, :].argsort()]
        split_by_index = [np.median(array) for array in np.split(both_sorted[:, 1], np.unique(both_sorted[:, 0], return_index=True)[1][1:])]
        metrics = MetricsHelper(truth, masks={target: mask}, pred_cutoff=0.7)
        metrics.update(np.asarray(split_by_index), target)

        return metrics.get_metrics(*self.metrics)
