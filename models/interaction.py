import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from .model_utils import get_activation

class DNN(nn.Module):
    def __init__(self, in_dim, layer_dim, dropout_rate=.0, l2_reg=.0, use_bn=False, act="relu",
                 device="cuda", dice_dim=3):
        """
        DNN模块初始化
        :param in_dim: 标量,输入tensor的维度
        :param layer_dim: 数组, 描述DNN层级
        :param dropout_rate: dropout参数
        :param l2_reg: L2正则化
        :param use_bn: 是否使用批标准化
        :param act: 激活函数
        :param device: cpu/gpu
        :param dice_dim: dice_dim
        """
        super(DNN, self).__init__()
        self.use_bn = use_bn
        assert len(layer_dim) > 0, "hidden layers should not be empty"
        layer_dim = (in_dim,) + tuple(layer_dim)
        self.act_layers = nn.ModuleList([get_activation(act) if act.lower() != 'dice' else
                                         get_activation(act, layer_dim[i + 1], dice_dim) for i in
                                         range(len(layer_dim) - 1)])
        self.l2 = torch.FloatTensor(np.array(l2_reg))

        self.dropout = nn.Dropout(dropout_rate)
        self.linears = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i + 1]) for i in range(len(layer_dim) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(layer_dim[i + 1]) for i in range(len(layer_dim) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(tensor)
        self.device = device
        #return module
        self.to(device)

    def forward(self, inputs):
        deep_inputs = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_inputs)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.act_layers[i](fc)
            fc = self.dropout(fc)
            deep_inputs = fc
        return deep_inputs

    def reg_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for module in self.linears:
            for name, parameter in module.named_parameters():
                if 'weight' in name:
                    total_reg_loss += (self.l2 * torch.sum(torch.square(parameter)))
        return total_reg_loss