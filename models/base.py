import logging
from six import add_metaclass
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.optim import AdamW
from .hooks.hook import Hook


@add_metaclass(ABCMeta)
class BaseModule(nn.Module):
    """Base module for all modules in deep scale free."""
    def __init__(self, optimizer, lr):
        """
        Initialize BaseModule, inherited from `torch.nn.Module`
        :param optimizer: The optimizer of model(obj: 'torch.optim.Optimizer').
        :param lr: The learning rate of optimizer.
        """
        super(BaseModule, self).__init__()
        self._hooks = []
        self._epochs = 0
        self._iter = 0
        self._output = {}
        self.best_metric = -1.0
        self.lr = lr
        # self.optimizers = optimizer
        self.optimizername = optimizer
        self.optimizer = None
        self.lr_scheduler_fc = None

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epochs

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def output(self):
        """int: Maximum training epochs."""
        return self._output

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def register_hook(self, hook):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
        """
        assert isinstance(hook, Hook)

        # insert the hook to a sorted list
        self._hooks.insert(0, hook)
    
    @abstractmethod
    def forward_train(self, *data, **kwargs):
        """
        :param data: Most of the cases are labels and input data, and sometimes there has weights.
        :param kwargs: Specific to concrete implementation.
        :return: 
        """
        raise NotImplemented

    @abstractmethod
    def forward_test(self, *data, **kwargs):
        """
        :param data: Most of the cases are labels and input data.
        :param kwargs: Specific to concrete implementation.
        :return: 
        """
        raise NotImplemented

    def forward(self, *data, return_loss=False, **kwargs):
        """Calls either 'forward_train' or 'forward_test' depending on whether 'return_loss' is 'True'."""
        if return_loss:
            return self.forward_train(*data, **kwargs)
        else:
            return self.forward_test(*data, **kwargs)

    @abstractmethod
    def train_epoch(self, data_loader):
        raise NotImplemented

    @abstractmethod
    def eval_epoch(self, data_loader):
        """This method defines an epoch during evaluation."""
        raise NotImplemented

    def _get_optimizers(self, optimizer, lr):
        """
        :param optimizer: String (name of optimizer) or torch optimizer instance.
        :param lr: learning rate of optimizer.
        :return: A List of torch optimizer instances.
        """
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
            elif optimizer == "adamw":
                optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return [optim]

    def freeze(self, layer_dim):
        """Some layers of the network can be selectively frozen."""
        pass

    def to_dist(self, device_ids):
        """Distributed training."""
        device_ids = device_ids if torch.cuda.is_available() else None
        model = torch.nn.parallel.DistributedDataParallel(
            self, device_ids=device_ids)
        model._get_optimizers = self._get_optimizers
        return model

    @abstractmethod
    def save_checkpoint(self, save_dir, scale_free, is_chief, rank):
        """Save the checkpoint of model"""
        pass

