# encoding: utf-8
# import sys, os
#
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)

import sys, os
import numpy as np
import torch
import logging
from abc import abstractmethod

from .base import BaseModule
from utils import metric
from transformers import get_scheduler
from tensorboardX import SummaryWriter

class BaseRank(BaseModule):
    def __init__(self, optimizer, lr, metrics_name=(metric.names.AUC,metric.names.gAUC, metric.names.LOGLOSS),
                 best_metric_name=metric.names.AUC, user_ids_feature_name=None):
        super(BaseRank, self).__init__(optimizer, lr)
        self.user_ids_feature_name = user_ids_feature_name
        self.best_metric_name = best_metric_name
        self.metrics = metric.get_metrics(metrics_name)

        # 新增：tensorboard writer
        self.writer = SummaryWriter(logdir='./runs')
        self.global_step = 0

    @abstractmethod
    def compute_predictions(self, *data, **kwargs):
        raise NotImplemented
    
    @abstractmethod
    def compute_loss(self, *data, **kwargs):
        raise NotImplemented
    
    def forward_train(self, *data, **kwargs):
        predictions = self.compute_predictions(*data)
        total_loss = self.compute_loss(*data, predictions)
        return total_loss

    def forward_test(self, *data, **kwargs):
        predictions = self.compute_predictions(*data)
        return predictions

    def train_epoch(self, data_loader):
        """
        This method defines an epoch during training, the whole process including forward propagation,
        backward propagation adn optimizer updating. And the loss will be added to the hook.
        :param data_loader: The dataloader of dataset.
        :return: The loss of one epoch.
        """
        import tqdm
        self.train()
        loss_epoch = 0
        self.call_hook('before_train_epoch')

        if self.optimizer == None:
            self.optimizer = self._get_optimizers(self.optimizername, self.lr)[0]
            self.lr_scheduler_fc = get_scheduler(
                name="linear", optimizer=self.optimizer, num_warmup_steps=20,
                num_training_steps=200
            )
        self._epochs += 1
        #optimizers = self._get_optimizers(self.optimizers, self.lr)
        for iter_step, data in tqdm.tqdm(enumerate(data_loader)):
            self._iter = iter_step
            self.call_hook('before_train_iter')
            loss = self.forward(data, return_loss=True)
            self.output.clear()
            self.output['loss'] = loss
            self.call_hook('after_train_iter')
            # for opt in optimizers:
            #     opt.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            # for opt in optimizers:
            #     opt.step()
            self.optimizer.step()
            loss_epoch += loss.item()
            # if iter_step % 1000 == 0:
            #     logging.info("iter loss {}".format(loss.item()))
            # 新增：记录loss和lr
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.global_step += 1

        self.call_hook('after_train_epoch')
        self.lr_scheduler_fc.step()
        print("loss {}, lr {}".format(loss_epoch / len(data_loader), self.optimizer.param_groups[0]['lr']))
        return loss_epoch / len(data_loader)

    def eval_epoch(self, data_loader):
        import tqdm
        self.eval()
        pred_ans = []
        label_ans = []
        user_ids_ans = []
        #self.call_hook('before_val_epoch')
        for iter_step, data in tqdm.tqdm(enumerate(data_loader)):
            # self._iter = iter_step
            with torch.no_grad():
                y_pred = self.forward(data, return_loss=False).to('cpu').numpy()
                labels = data['label'].cpu().numpy()
                
                pred_ans.append(y_pred)
                label_ans.append(labels)
                
                # gAUC input
                if self.user_ids_feature_name:
                    #user_ids = batch_labels_and_weights[:, -1].cpu().numpy()
                    user_ids = data[self.user_ids_feature_name].cpu().numpy()
                    user_ids_ans.append(user_ids)

        # TODO: do allgather to make sure all devices get same evaluation results
        pred = np.concatenate(pred_ans).astype("float64")
        label = np.concatenate(label_ans).astype("float64")
        
        self.output.clear()
        for metric_name, metric in self.metrics.items():
            # gAUC
            if str(metric_name).lower() == 'gauc':
                user_id = np.concatenate(user_ids_ans).astype("float64")
                self.output[metric_name] = metric(label, pred, user_id)
            else:
                self.output[metric_name] = metric(label, pred)
            # 新增：记录到tensorboard
            self.writer.add_scalar(f'eval/{metric_name}', self.output[metric_name], self.global_step)
            print("{} : {}".format(metric_name, self.output[metric_name]))

        # logger print
        #self.call_hook('after_val_epoch')


        
    def _get_optimizers(self, optimizer, lr):
        if isinstance(optimizer, str):
            # adam need to be overridden
            if optimizer == "adam":
                # if not self.embedding.emb.sparse:
                #     optim = [torch.optim.Adam(params=self.parameters(), lr=lr)]
                # else:
                #     optim = [torch.optim.SparseAdam(params=self.parameters(), lr=lr)]
                optim = [torch.optim.Adam(params=self.parameters(), lr=lr)]
            if optimizer == "adamw":
                optim = [torch.optim.AdamW(params=self.parameters(), lr=lr)]
            elif optimizer == "adagrad":
                optim = [torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)]
            elif optimizer == "sgd":
                optim = [torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)]
            else:
                raise NotImplementedError
        else:
            optim = [optimizer]
        return optim

    def save_checkpoint(self, save_dir, scale_free, is_chief, rank, incremental_url=None):
        from moxing.framework.file import file_io
        if self.output[self.best_metric_name] > self.best_metric:
            self.best_metric = self.output[self.best_metric_name]
            if save_dir:
                logging.info('saving model for epoch=%d' % self.epoch)
                if scale_free or (is_chief and not scale_free):
                    state_dict = self.state_dict()
                    torch.save(state_dict,
                               os.path.join(save_dir, 'epoch%d_rank%d.pth' % (self.epoch, rank)))
                    file_io.copy(os.path.join(save_dir, 'epoch%d_rank%d.pth' % (self.epoch, rank)),
                                 os.path.join(save_dir, 'model', 'best.pth'))
                    # incremental train
                    if incremental_url:
                        file_io.copy(os.path.join(save_dir, 'model', 'best.pth'),
                                     os.path.join(incremental_url, 'best.pth'))

    @staticmethod
    def _get_labels_and_weights(labels_and_weights, device, use_weight=False):
        if labels_and_weights is None:
            raise ValueError('labels not found when loss is required.')
        labels = labels_and_weights['label']

        if use_weight and 'weight' in labels_and_weights.keys():
            weights = torch.unsqueeze(labels_and_weights['weight'], 1)
            weights = weights.to(device)
        elif use_weight and 'weight' not in labels_and_weights.keys():
            raise ValueError('weight not found when is required.')
        else:
            weights = None

        labels = torch.unsqueeze(labels, 1).to(device)

        return labels, weights
