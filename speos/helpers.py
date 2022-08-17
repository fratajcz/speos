import numpy as np
import torch
import os


class Companion:
    """Abstract class for all classes that accompany and manage the training process"""
    def __init__(self, mode="min", patience=10, min_impr=1e-3, limit=1e-8, mean_ws=2):
        self.mode = mode
        self.patience = patience
        self._init_patience = patience
        self.min_impr = min_impr
        self.limit = limit
        self.mean_ws = mean_ws

        self.hits = 0
        self.history = []
        self._raw_history = []
        self.iteration = 0
        self._init_best_value = -1e10 if self.mode == "max" else 1e10
        self.best_value = self._init_best_value

    def step(self):
        raise NotImplementedError

    def has_improved(self, value):
        if self.mode == "min":
            return value < self.best_value
        else:
            return value > self.best_value

    def hits_limit(self, new_value):
        if self.mode == "min":
            return new_value < self.limit
        else:
            return new_value > self.limit


class LRScheduler(Companion):
    def __init__(self, optimizers,
                 mode: str = "min",
                 factor: float = 0.33,
                 patience: int = 10,
                 min_impr: float = 1e-3,
                 limit: float = 1e-8,
                 mean_ws: int = 2):
        super(LRScheduler, self).__init__(mode=mode, patience=patience, min_impr=min_impr, limit=limit, mean_ws=mean_ws)
        self.factor = factor
        self.optimizers = optimizers

    def step(self, value):
        self.history.append(np.mean(self.history[-self.mean_ws:] + [value]))
        self._raw_history.append(value)

        if self.has_improved(value):
            self.best_value = self.history[-1]
            self.patience = self._init_patience
        else:
            self.patience -= 1

        if self.patience <= 0:
            new_lr = self.adjust_lr()
            self.patience = self._init_patience
            self.best_value = self._init_best_value
            return new_lr

        else:
            return None

    def adjust_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * self.factor
                if new_lr >= self.limit:
                    param_group['lr'] = new_lr
                else:
                    param_group['lr'] = new_lr = self.limit

        return new_lr


class CheckPointer(Companion):
    def __init__(self, model, path="./models/model",
                 mode: str = "min",
                 min_impr: float = 0,
                 verbose: bool = True):
        super(CheckPointer, self).__init__(mode=mode, min_impr=min_impr)
        self.model = model
        self.path = path + ".pt"

        if self.path:
            try:
                os.stat(os.path.dirname(self.path))
            except FileNotFoundError:
                os.makedirs(os.path.dirname(self.path))

    def step(self, value):
        self._raw_history.append(value)
        self.history = self._raw_history[:]

        if self.has_improved(value):
            self.save(value)
            self.best_value = self._raw_history[-1]
        elif not os.path.exists(self.path):
            # save model on the first invocation of step() (i.e. if the model has not been saved at all yet)
            self.save(value)

    def save(self, value):
        self.state_dict = {
            'epoch': len(self._raw_history) - 1,
            'model_state_dict': self.model.state_dict(),
            'value': value
        }
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        torch.save(self.state_dict, self.path)

    def load(self, path=None):
        '''loads a state dict either from the current run or from a given path'''
        if path is None:
            path = self.path
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            # in case there is no cuda available, always load to cpu
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.state_dict = state_dict

    def restore(self, path=None):
        '''restores the model to the state that it was in previously or from a given path'''
        self.load(path)
        self.model.load_state_dict(self.state_dict["model_state_dict"])
        return self.state_dict["epoch"], self.state_dict["value"]


class EarlyStopper(Companion):
    def __init__(self, patience=50,
                 mode: str = "min",
                 min_impr: float = 0,
                 verbose: bool = True):
        super(EarlyStopper, self).__init__(patience=patience, mode=mode, min_impr=min_impr)
        self.verbose = verbose

    def step(self, value) -> bool:
        """Returns True if training did not improve in the last n epochs"""
        self._raw_history.append(value)
        self.history = self._raw_history[:]

        if self.has_improved(value):
            self.patience = self._init_patience
            self.best_value = self._raw_history[-1]
        else:
            self.patience -= 1

        if self.patience <= 0:
            stop_training = True
        else:
            stop_training = False

        return stop_training
