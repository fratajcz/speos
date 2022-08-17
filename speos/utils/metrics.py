import numpy as np
from sklearn import metrics


class MetricsHelper:
    def __init__(self, y_truth, pred_cutoff: float, masks=None, input_type="prob"):
        self.y_truth = y_truth.squeeze()
        self.output = None
        self.masks = {"all": np.ones_like(y_truth).astype(np.bool8)}
        self.update_masks(masks)
        self.input_type = input_type
        assert self.input_type in ["prob", "logit"]
        self.pred_cutoff = pred_cutoff

    def update(self, output: np.ndarray, mask_key: str):
        self.mask_key = mask_key

        if output.shape[0] == self.masks["all"].shape[0]:
            self.output = output.squeeze()
        else:
            raise ValueError("Shape of output {} does not fit shape of masks {}. Check model output shape.".format(output.shape, self.masks["all"].shape))

    def update_masks(self, masks):
        if masks is None:
            return

        for key in masks.keys():
            masks[key] = masks[key].squeeze()

        self.masks.update(masks)

    def get_truth(self):
        return (self.y_truth[self.masks[self.mask_key]]).squeeze()

    def get_output(self):
        return (self.output[self.masks[self.mask_key]]).squeeze()

    @property
    def prediction(self) -> np.ndarray:
        return (self.probability > self.pred_cutoff).astype(np.uint8)

    @property
    def probability(self) -> np.ndarray:
        # in case output is already between 0 and 1 (in case of mse)
        # if self.output.min() >= 0 and self.output.max() <= 1:
        if self.input_type == "prob":
            return self.get_output()
        else:
            return 1 / (1 + np.exp(-self.get_output()))

    @property
    def auprc(self):
        return metrics.average_precision_score(self.get_truth(), self.get_output())

    @property
    def auroc(self):
        try:
            results = metrics.roc_auc_score(self.get_truth(), self.get_output())
        except ValueError:
            results = np.nan

        return results

    @property
    def accuracy(self):
        return np.sum(np.equal(self.get_truth(), self.prediction)) / len(self.get_truth())

    @property
    def recall(self):
        confusion = self.confusion
        try:
            return confusion[1][1] / np.sum(confusion[1])
        except IndexError:
            return np.nan

    @property
    def au_rank_cdf(self):
        """ Inspired by http://arxiv.org/abs/1504.06837 is basically AUROC without false positives """

        ranks = np.asarray([i for j in self.rs_filtered for i in j])
        ranks_plus_stops = np.concatenate(([0], ranks, [self.y_truth.shape[0]]), axis=0)
        step_size = np.diff(ranks_plus_stops)
        total_positives = ranks.shape[0]
        tpr = np.asarray([(i + 1) / total_positives for i in range(ranks.shape[0])])
        cdf = np.repeat(np.concatenate(([0], tpr), axis=0), step_size, axis=0)
        if self.mask_key == "all":
            valid_places = np.sum(self.y_truth[self.masks[self.mask_key]])
        else:
            valid_places = np.sum(self.masks[self.mask_key])
        cdf_filtered = cdf[:valid_places]
        au_cdf = cdf_filtered.mean()
        """
        ranks = self.rs_filtered[0]
        legal_ranking = np.zeros_like(self.y_truth)
        legal_ranking[ranks] = 1
        total_positives = np.sum(legal_ranking)
        num_relevant_at_k = np.tril(np.repeat(legal_ranking.reshape(1, -1), legal_ranking.shape[0], 0)).sum(axis=1)
        rank_cdf = num_relevant_at_k / total_positives
        au_cdf = rank_cdf.sum() / legal_ranking.shape[0]
        """
        return au_cdf

    @property
    def precision(self):
        confusion = self.confusion
        try:
            return confusion[1][1] / np.sum((confusion[1][1], confusion[0][1], 1e-10))
        except IndexError:
            return np.nan

    @property
    def confusion(self):
        return metrics.confusion_matrix(self.get_truth(), self.prediction)

    @property
    def f1(self):
        return metrics.f1_score(self.get_truth(), self.prediction)

    @property
    def mrr_raw(self):
        return self._mrr([self.rs_raw])

    @property
    def mrr_filtered(self):
        return self._mrr([self.rs_filtered])

    @property
    def mean_rank_raw(self):
        return self._mean_rank([self.rs_raw])

    @property
    def mean_rank_filtered(self):
        return self._mean_rank([self.rs_filtered])

    @property
    def hits_at_100_raw(self):
        return self.hits_at_k_raw(100)

    @property
    def hits_at_100_filtered(self):
        return self.hits_at_k_filtered(100)

    def hits_at_k_raw(self, k):
        return self._hits_at_k([self.rs_raw], k)

    def hits_at_k_filtered(self, k):
        return self._hits_at_k([self.rs_filtered], k)

    def _mean_rank(self, rs):
        return np.mean([r + 1 for sublist in rs for r in sublist])

    def _hits_at_k(self, rs, k):
        return np.mean([1 if r < k else 0 for sublist in rs for r in sublist])

    def _mrr(self, rs):
        return np.mean([1. / (r + 1) for sublist in rs for r in sublist])

    def ordered_truth_(self, mask_key="all"):
        ranking = np.argsort(self.output)[::-1]
        # generate a subset truth that has same shape as ground truth but contains only the 1s of train/val/test/all
        subset_truth = self.y_truth.copy()
        mask = self.masks[mask_key]
        subset_truth[~mask] = 0
        return subset_truth[ranking]

    @property
    def ordered_truth(self):
        return self.ordered_truth_(self.mask_key)

    @property
    def ordered_truth_train(self):
        return self.ordered_truth_("train")

    @property
    def ordered_truth_val(self):
        return self.ordered_truth_("val")

    @property
    def ordered_truth_test(self):
        return self.ordered_truth_("test")

    @property
    def ordered_truth_all(self):
        return self.ordered_truth_("all")

    @property
    def rs_raw(self):
        if len(self.ordered_truth.shape) > 1:
            return list(np.asarray(r).nonzero()[0] for r in self.ordered_truth if np.sum(r) > 0)
        elif len(self.ordered_truth.shape) == 1:
            return list(np.asarray(self.ordered_truth).nonzero()[0])

    @property
    def rs_filtered(self):
        additional_truth = self.ordered_truth_all - self.ordered_truth

        return list(np.asarray(r).nonzero()[0] for r in self.filter(self.ordered_truth, additional_truth) if np.sum(r) > 0)

    def filter(self, rs, additional_truth):
        total_before = np.sum(rs)
        self.filtered_entities = []
        rs_filtered = []
        if additional_truth is not None:
            # if True:
            # also remove all known true examples from the other sets
            rs_prefiltered = []
            to_delete = additional_truth.nonzero()[0]
            rs_prefiltered.append(np.delete(rs, to_delete))
            self.filtered_entities.append(to_delete)
            total_after = np.sum(np.sum(r) for r in rs_prefiltered)
            assert total_before == total_after  # nothing lost filtering for out-of-sample edges
            rs = rs_prefiltered

        for r in rs:
            while np.sum(r) > 0:
                best = r.nonzero()[0][0]
                best_r = np.zeros_like(r)
                best_r[best] = 1
                rs_filtered.append(best_r)
                r = np.delete(r, best)

        total_after = np.sum(np.sum(r) for r in rs_filtered)
        assert total_before == total_after  # nothing lost in filtering
        assert len(rs_filtered) == total_before  # every edge gets its own array
        for r in rs_filtered:
            assert np.sum(r) == 1  # only one edge in every array

        return rs_filtered

    def get_metrics(self, *args):
        for arg in args:
            return [self.process_function(function) for function in args]

    def process_function(self, function_string):
        if "_at_" in function_string:
            function = function_string.split("_")[0]
            k = function_string.split("_")[-2]
            suffix = function_string.split("_")[-1]
            total_function_string = "self.{}_at_k_{}({})".format(function, suffix, k)
            try:
                result = eval(total_function_string)
            except ValueError:
                return np.nan
        else:
            try:
                result = getattr(self, function_string)
            except ValueError:
                return np.nan

        return result
