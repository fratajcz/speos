import numpy as np
import os

from speos.explanation import InputExplainer
from speos.preprocessing.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.helpers import EarlyStopper, LRScheduler, CheckPointer
from speos.utils.logger import setup_logger
from speos.utils.metrics import MetricsHelper
from speos.utils.datahandlers import ResultsHandler
from speos.models import ModelBootstrapper

import torch
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter


class Experiment:
    """ Puts most puzzle pieces together and has everything to execute training """

    def __init__(self, config, resultshandler=None, id=None):
        torch.set_default_dtype(torch.float64)

        self.config = config
        self.config.save()
        self.name = self.config.name
        self.logger_name = __name__ + " ({})".format(id) if id is not None else __name__

        logger = setup_logger(config, self.logger_name)

        if not os.path.exists(os.path.dirname(self.config.model.plot_dir)):
            os.makedirs(os.path.dirname(self.config.model.plot_dir))

        logger.info("Starting run {}".format(self.name))
        logger.info("Using device(s): {}".format(self.devices))

        self.dataset = DatasetBootstrapper(holdout_size=config.input.holdout_size, name=self.name, config=self.config).get_dataset()

        node_data = self.dataset.data

        if self.config.model.architecture == "LINKX":
            input_dim = (node_data.x.shape[1], node_data.x.shape[0])
        else:
            input_dim = node_data.x.shape[1]

        self.model = ModelBootstrapper(
            config, input_dim, self.dataset.num_relations).get_model()

        logger.info(self.model.architectures[-1])

        self.model = self.model.to(self.devices[0])

        if self.config.crossval.mode == "complex":
            n_folds = self.config.crossval.n_folds * \
                (self.config.crossval.n_folds + 1)
        else:
            n_folds = self.config.crossval.n_folds
        
        self.resultshandler = ResultsHandler(file_path=os.path.join(self.config.inference.save_dir, self.config.name + ".h5"),
                                             n_folds=n_folds,
                                             shape=node_data.x.shape,
                                             explanation=self.config.inference.input_explain,
                                             index=self.dataset.node_df["hgnc"].tolist()) \
            if resultshandler is None else resultshandler

        if resultshandler is None:
            logger.info("Created new ResultsHandler pointing to {}".format(
                self.resultshandler.file_path))
        else:
            logger.info("Using existing ResultsHandler pointing to {}".format(
                self.resultshandler.file_path))

        logger.info("Received data with {} train positives, {} train negatives, {} val positives, {} val negatives, {} test positives and {} test negatives".format(
            *[truth[mask].sum().long().item() for mask in [node_data.train_mask, node_data.val_mask, node_data.test_mask] for truth in [node_data.y, 1 - node_data.y]]))

        input_type = "logit" if self.config.model.loss == "bce" else "prob"

        self.metrics = MetricsHelper(node_data.y.detach().cpu().numpy(),
                                     pred_cutoff=config.eval.cutoff,
                                     masks={"train": node_data.train_mask.detach().cpu().numpy(),
                                            "val": node_data.val_mask.detach().cpu().numpy(),
                                            "test": node_data.test_mask.detach().cpu().numpy()},
                                     input_type=input_type)

        self.data = self.dataset.data.to(self.devices[0])

        if self.model.requires_sgd:
            self.explainer = InputExplainer(
                node_data, self.model, self.dataset.preprocessor.get_feature_names(), config)

            self.scheduler = LRScheduler(self.model.optimizers, mode=config.scheduler.mode,
                                         factor=config.scheduler.factor, patience=config.scheduler.patience, limit=config.scheduler.limit)

            self.earlystopper = EarlyStopper(patience=self.config.es.patience, mode=self.config.es.mode)
            self.max_epochs = config.training.max_epochs
        else:
            self.max_epochs = 1

        self.checkpointer = CheckPointer(
                self.model, self.config.model.save_dir + self.name, mode=config.es.mode)


        self.n_neighbors = []

        self.tensorboard = True

        self.train_precision = None
        self.out = None

    def run(self, epochs=None):
        if self.tensorboard:
            self.tbpath = os.path.join('./runs', self.name)
            logger = setup_logger(self.config, self.logger_name)
            logger.info(
                "Writing TensoBoard data to {}".format(self.tbpath))
            self.writer = SummaryWriter(self.tbpath, comment="test")

        run_epochs = self.max_epochs if epochs is None else epochs

        for epoch in range(run_epochs):
            self.epoch = epoch

            logger.info("Epoch {}".format(epoch))
            self.train()

            if torch.sum(self.data.val_mask) > 0:
                target = "val"
            else:
                target = "train"

            self.eval(target)

            self.checkpointer.step(self.checkpoint_on)
            if self.model.requires_sgd:
                new_lr = self.scheduler.step(self.checkpoint_on)
                stop_training = self.earlystopper.step(self.checkpoint_on)

                if new_lr is not None:
                    logger.info(
                        "Adjusted Learning rate to {}.".format(new_lr))
                if stop_training:
                    logger.info(
                        "Reached Early Stopping Criterion at epoch {}, aborting training".format(epoch))
                    break

        # roll back to the best epoch before plotting
        rollback_epoch, rollback_value = self.checkpointer.restore()

        logger.info("Rolled back model to epoch {} with value {}".format(
            rollback_epoch, rollback_value))

        if self.config.model.plot:
            self.eval_and_plot()

    def train(self):
        """
            Runs one training step.
        """
        self.train_out, loss = self.model.step(self.data, self.data.train_mask)

        self.writer.add_scalar('Loss/train', loss, self.epoch)

        if self.model.requires_sgd:
        # monitor LR decay
            for i, optimizer in enumerate(self.model.optimizers):
                for param_group in optimizer.param_groups:
                    lr = float(param_group['lr'])
                self.writer.add_scalar(
                    'Parameters/lr #{}'.format(i), lr, self.epoch)

        # TODO: make this handle single and multi output variants
        self.metrics.update(
            self.train_out[-1, :].squeeze().detach().cpu().numpy(), "train")

        probability, auprc, self.train_precision = self.metrics.get_metrics(
            "probability", "auprc", "precision")

        self.writer.add_scalar('Paramerers/AUPRC train', auprc, self.epoch)

        if self.tensorboard and self.epoch % 50 == 0:
            self.document_distributions(probability, self.data.y[self.data.train_mask].detach(
            ).cpu().bool(), title="train Discriminator")
            class_predictions = torch.sigmoid(
                self.train_out[0, :].squeeze()).detach().cpu().numpy()
            self.document_distributions(class_predictions, torch.zeros_like(
                self.train_out[0, :].squeeze()).detach().cpu().bool(), title="train Classifier")

        logger = setup_logger(self.config, self.logger_name)
        logger.info("Training Loss: {}".format(loss))

    def eval(self, target="val"):
        """
            Runs one evaluation step.
        """
        if target == "val":
            mask = self.data.val_mask
        elif target == "train":
            mask = self.data.train_mask
        elif target == "test":
            mask = self.data.test_mask
        elif target == "all":
            mask = torch.ones_like(self.data.test_mask, dtype=torch.bool)
        else:
            raise ValueError

        with torch.no_grad():
            self.val_out, loss = self.model.step(
                self.data, mask, eval_flag=True)

        if self.tensorboard:
            self.writer.add_scalar('Loss/eval', loss, self.epoch)

        self.metrics.update(
            self.val_out[-1, :].squeeze().detach().cpu().numpy(), target)

        probability, prediction, accuracy, recall, precision, auroc, auprc, f1, mrr, mr = self.metrics.get_metrics(
            "probability", "prediction", "accuracy", "recall", "precision", "auroc", "auprc", "f1", "mrr_filtered", "mean_rank_filtered")

        self.recall = recall
        # if val set has only positives, use composite f1, else use normal f1
        if self.data.y[self.data.val_mask].sum() == self.data.val_mask.sum() and self.config.es.metric != "mean_rank_filtered":
            self.checkpoint_on = 2 * (self.train_precision * self.recall) / (
                self.train_precision + self.recall) if self.train_precision is not None else self.recall
        else:
            self.checkpoint_on = self.metrics.get_metrics(
                self.config.es.metric)[0]

        if self.tensorboard:
            self.writer.add_scalar('Paramerers/AUPRC eval', auprc, self.epoch)
            self.writer.add_scalar(
                'Paramerers/{} eval'.format(self.config.es.metric), self.checkpoint_on, self.epoch)

        logger = setup_logger(self.config, self.logger_name)
        logger.info("Eval Loss: {}, Accuracy: {}, Recall: {}, Precision: {}, AUROC: {}, AUPRC: {}, F1: {}, MRR: {}, MR: {}, Target: {}".format(
            loss, accuracy, recall, precision, auroc, auprc, f1, mrr, mr, target))

        return self.data.y[mask].detach().cpu().numpy().astype(np.uint8), prediction, probability

    def document_distributions(self, predictions, labels, title):

        positive_predictions = predictions[labels]
        negative_predictions = predictions[~labels]

        try:
            if len(positive_predictions) > 0:
                self.writer.add_histogram("Predictions for Positive Examples ({})".format(
                    title), positive_predictions, self.epoch)
            if len(negative_predictions) > 0:
                self.writer.add_histogram("Predictions for Negative Examples ({})".format(
                    title), negative_predictions, self.epoch)
        except ValueError:
            pass

    def eval_and_plot(self):
        for name, mask in zip(("train", "val"), [self.data.train_mask, self.data.val_mask]):
            # skip empty splits
            if torch.sum(mask) == 0:
                continue

            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

            self.eval(target=name)
            self.metrics.update(
                self.val_out[-1, :].squeeze().detach().cpu().numpy(), name)

            pred_proba = self.metrics.get_metrics("probability")[0]

            class_proba = [
                "Mendelian" if y > 0.5 else "Unknown" for y in self.data.y[mask].detach().cpu().numpy().squeeze()]

            sns.violinplot(y=pred_proba, x=class_proba, cut=0)

            try:
                os.makedirs(self.config.model.plot_dir)
            except FileExistsError:
                pass

            fig.savefig(os.path.join(self.config.model.plot_dir,
                        "violin_{}_{}.png".format(name, self.name)))
            fig.clf()

    def balance_losses(self, losses):
        """deprectaed, moved into the model class"""
        positives = torch.nonzero(self.data.y[self.data.train_mask])
        negatives = torch.nonzero(1 - self.data.y[self.data.train_mask])

        dilution = 1 if self.config.training.dilution is None else self.config.training.dilution

        if dilution == "max" or (positives.shape[0] * dilution) >= negatives.shape[0]:
            sampled_negatives = negatives
            dilution = negatives.shape[0] / positives.shape[0]
        else:
            num_to_sample = positives.shape[0] * dilution
            sample_indices = torch.randint(
                0, negatives.shape[0], (num_to_sample,)).long()

        sampled_negatives = negatives[sample_indices]
        pos_loss = losses[positives] * self.config.training.pos_weight
        neg_loss = losses[sampled_negatives] / dilution

        loss = torch.mean(torch.cat((pos_loss, neg_loss), dim=0))

        return loss

    @property
    def devices(self):
        cuda = self.config.cuda
        logger = setup_logger(self.config, self.logger_name)
        if cuda:
            try:
                logger.info("Cuda is available: {}".format(
                    torch.cuda.is_available()))
                assert torch.cuda.is_available()
            except AssertionError as e:
                if cuda == "auto":
                    logger.info(
                        "CUDA set to auto, no CUDA device detected, setting to CPU")
                    devices = ["cpu"]
                    return devices
                else:
                    logger.error(
                        "Specified that job should be run on CUDA, but no CUDA devices are available. Aborting...")
                    raise e

            try:
                available_cuda_devices = ["cuda:{}".format(
                    device) for device in range(torch.cuda.device_count())]
                if cuda is True:
                    devices = available_cuda_devices
                elif cuda == "auto":
                    devices = available_cuda_devices
                elif type(cuda) == str:
                    devices = [cuda]
                elif type(cuda) == list:
                    devices = cuda

                for device in devices:
                    assert type(device) == str
                    assert device.startswith("cuda:")
                    assert device in available_cuda_devices

            except AssertionError:
                logger.error("Specified cuda device(s) {} not in available cuda device(s): {}. Check Spelling or Numbering".format(
                    cuda, available_cuda_devices))
                raise ValueError("Specified cuda device(s) {} not in available cuda device(s): {}. Check Spelling or Numbering".format(
                    cuda, available_cuda_devices))
        else:
            logger.info(
                "CUDA is set to {}, using cpu as fallback".format(cuda))
            devices = ["cpu"]

        return devices


class InferenceEngine(Experiment):
    def __init__(self, config, resultshandler=None):
        super(InferenceEngine, self).__init__(
            config, resultshandler=resultshandler)

        if config.inference.gnn_explain:
            from torch_geometric.nn import GNNExplainer
            self.gnnexplainer = GNNExplainer(
                self.model, epochs=200, return_type='log_prob')
            # TODO: use in infer()
        else:
            self.gnnexplainer = None

        self.input_explain = config.inference.input_explain
        self.threshold = config.inference.cutoff_value
        self.tensorboard = True
        self.epoch = None
        logger = setup_logger(self.config, self.logger_name)

        if self.tensorboard:
            self.tbpath = os.path.join('./inference', self.name)
            logger.info(
                "Writing TensoBoard data to {}".format(self.tbpath))
            self.writer = SummaryWriter(self.tbpath, comment="test")

        if self.config.inference.save or self.config.inference.save_sorted:
            try:
                os.stat(self.config.inference.save_dir)
            except FileNotFoundError:
                os.makedirs(self.config.inference.save_dir)

    def infer(self, plot=True):
        if self.epoch is None:
            self.restore_model()

        logger = setup_logger(self.config, self.logger_name)

        logger.info("Starting Inference.")

        if self.input_explain:
            logger.info("Calculating Explanations.")
            attributions, explanations_plot = self.explainer.explain(plot=plot)
            for i in range(attributions.shape[1]):
                self.writer.add_histogram(
                    'Deeplift (positive predicted unknowns)', attributions[:, i], i)
            if explanations_plot is not None:
                self.writer.add_images('Input Explanations', np.stack(
                    explanations_plot, axis=0), 0)
            logger.info("Finished.")

        self.tensorboard = False
        truth, prediction, probability = self.eval(
            self.config.inference.target)

        node_df = self.dataset.node_df
        columns_to_drop = ["truth", "train", "val", "test"]
        for column in columns_to_drop:
            try:
                node_df.drop(column, axis=1, inplace=True)
            except KeyError:
                continue
        results_df = pd.DataFrame(data=truth, columns=["truth"])
        results_df["prediction"] = prediction
        results_df["probability"] = probability
        results_df["train"] = self.data.train_mask.detach(
        ).cpu().numpy().astype(np.int8)
        results_df["val"] = self.data.val_mask.detach(
        ).cpu().numpy().astype(np.int8)
        results_df["test"] = self.data.test_mask.detach(
        ).cpu().numpy().astype(np.int8)
        self.resultshandler.add_results(results_df, name=self.name)
        

        if self.input_explain:
            explain_df = pd.DataFrame(data=attributions.numpy(
            ), columns=self.dataset.preprocessor.get_feature_names())

        if self.config.inference.save:
            path = os.path.join(
                self.config.inference.save_dir, self.name + ".tsv")
            logger.info("Writing Inference data to {}".format(
                self.resultshandler.file_path))
            if self.config.inference.input_explain:
                self.resultshandler.add_explanations(
                    explain_df, name=self.name)
                df = node_df.join(results_df).join(explain_df)
            else:
                df = node_df.join(results_df)
            if self.config.inference.save_tsv:
                df.to_csv(path, sep="\t")
            """
            if self.config.inference.save_sorted:
                path = os.path.join(self.config.inference.save_dir, self.name + "_sorted.tsv")
                self.logger.info("Writing sorted Inference data to {}".format(path))
                df_sorted = df.sort_values(by="probability",axis=0, ascending=False)
                df_sorted.to_csv(path, sep="\t")
            """

        return df

    def restore_model(self):
        logger = setup_logger(self.config, self.logger_name)
        logger.info("Restoring trained Model for Inference")
        self.epoch, value = self.checkpointer.restore()
        logger.info(
            "Restored trained Model at Epoch {} with Value {}".format(self.epoch, value))
