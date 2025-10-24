# encoding: utf-8
# Copyright 2021 ModelArts Authors from Huawei Cloud. All Rights Reserved.
# https://www.huaweicloud.com/product/modelarts.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import logging
import functools

from torch import distributed as dist
from .hook import Hook


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


class TensorboardHook(Hook):
    @master_only
    def __init__(self, log_dir=None, logging_freq=10):

        super(TensorboardHook, self).__init__()

        assert (log_dir is not None), (
            "please set log_dir")

        self.logging_freq = logging_freq
        self.log_dir = os.path.join(log_dir, 'tf_logs')
        self._writer = None

    @property
    def writer(self):
        return self._writer

    @master_only
    def before_train_epoch(self, model):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                'the dependencies to use torch.utils.tensorboard '
                '(applicable to PyTorch 1.1 or higher)')

        self._writer = SummaryWriter(self.log_dir)

    @master_only
    def after_train_epoch(self, model):
        self._writer.close()

    @master_only
    def after_train_iter(self, model):
        for tag, val in model.output.items():
            if isinstance(val, str):
                self._writer.add_text(tag, val, model.iter)
            else:
                self._writer.add_scalar(tag, val, model.iter)

            if model.iter % self.logging_freq == 0:
                logging.info(
                    "train epoch: {}, batch_id: {}, {} is: {}".format(
                        model.epoch, model.iter, tag, val.item() if isinstance(val, torch.Tensor) else val))

    @master_only
    def after_val_iter(self, model):
        if model.iter % self.logging_freq == 0:
            for tag, val in model.output.items():
                logging.info(
                    "eval epoch: {}, batch_id: {}, {} is: {}".format(
                        model.epoch, model.iter, tag, val.item() if isinstance(val, torch.Tensor) else val))
                
    @master_only
    def after_val_epoch(self, model):
        for tag, val in model.output.items():
            logging.info(
                "eval epoch: {}, {} is: {}".format(
                    model.epoch, tag, val.item() if isinstance(val, torch.Tensor) else val))
