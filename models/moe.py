import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed

from .moe_layer import MoeLayer
from .base_rank import BaseRank
from .embedding import Embedding, NoWtsEmbedding
from .interaction import DNN


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets, weights=None):
        BCE_loss = F.binary_cross_entropy_with_logits(input=inputs, target=targets, weight=weights, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = 10 * alpha_t * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class BaseWDL(nn.Module):
    def __init__(self,
                 layer_dim,
                 deep_field_size,
                 wide_field_size,
                 l2_reg_dnn=1e-5,
                 l2_reg_linear=1e-5,
                 dnn_dropout=.0,
                 dnn_act="relu",
                 device="cpu"):
        super(BaseWDL, self).__init__()

        self.dnn = DNN(deep_field_size,
                       layer_dim, l2_reg=l2_reg_dnn, act=dnn_act,
                       device=device, dropout_rate=dnn_dropout, use_bn=False)
        self.wide = nn.Linear(wide_field_size, out_features=1, bias=False).to(device)
        self.dnn_final = nn.Linear(layer_dim[-1], 1, bias=False).to(device)
        self.deep_field_size = deep_field_size
        self.wide_field_size = wide_field_size
        self.l2_reg_dnn = torch.FloatTensor(np.array(l2_reg_dnn))
        self.l2_reg_linear = torch.FloatTensor(np.array(l2_reg_linear))
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.device = device
        self.to(device)

    def forward(self, deep_features, wide_features):
        deep_features = deep_features.to(self.device)
        wide_features = wide_features.to(self.device)

        # wide
        wide_inputs = torch.reshape(wide_features, shape=(-1, self.wide_field_size))
        linear_logits = self.wide(wide_inputs)

        # deep
        dnn_inputs = torch.reshape(deep_features, shape=(-1, self.deep_field_size))
        dnn_outputs = self.dnn(dnn_inputs)

        logits = 0.2 * linear_logits + 0.8 * self.dnn_final(dnn_outputs)
        #logits = 0.999 * linear_logits + 0.001 * self.dnn_final(dnn_outputs)
        predictions = torch.sigmoid(logits + self.bias)
        return predictions

    def reg_loss(self):
        reg_loss = self.dnn.reg_loss()
        reg_loss += (self.l2_reg_linear * torch.sum(torch.square(self.wide.weight)))
        reg_loss += (self.l2_reg_dnn * torch.sum(torch.square(self.dnn_final.weight)))
        return reg_loss


class BatchedNormalizer(nn.Module):
    """
    Unified batched normalizer that processes multiple features at once.
    All normalization methods are handled in a single forward pass for efficiency.
    """
    def __init__(self, norm_type: str, num_features: int, device: str):
        """
        Args:
            norm_type: one of 'zscore', 'log1p', 'reciprocal', 'null'
            num_features: number of features to normalize with this method
            device: device to place the normalizer on
        """
        super().__init__()
        self.norm_type = norm_type
        self.num_features = num_features
        
        # Only zscore needs learnable parameters
        if norm_type == 'zscore':
            # Use BatchNorm1d with num_features channels for batched processing
            # Each feature gets its own running_mean and running_var
            self.norm = nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)
        else:
            self.norm = None
        
        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        Apply normalization to batched features.
        
        Args:
            x: shape (batch_size, num_features) - stacked features of the same norm type
            
        Returns:
            normalized tensor of shape (batch_size, num_features)
            
        Note:
            - zscore: uses BatchNorm1d for learnable normalization
            - log1p: applies log(1+x) element-wise (batched operation)
            - reciprocal: applies 1/(1+x) element-wise (batched operation)
            - null: pass through without modification
        """
        if self.norm_type == 'zscore':
            # BatchNorm1d: (batch_size, num_features) -> (batch_size, num_features)
            return self.norm(x)
        elif self.norm_type == 'log1p':
            # Element-wise log1p, but operates on entire batch at once
            return torch.log1p(x)
        elif self.norm_type == 'reciprocal':
            # Element-wise reciprocal, but operates on entire batch at once
            return torch.reciprocal(x + 1.0)
        else:  # 'null' or None
            # Pass through without modification
            return x


class MOE(BaseRank):
    EmbeddingModule = NoWtsEmbedding

    def __init__(self,
                 layer_dim,
                 embedding_dim,
                 feat_config,
                 l2_reg_embedding=1e-5,
                 l2_reg_dnn=1e-5,
                 l2_reg_linear=1e-5,
                 embedding_init_std=0.0001,
                 dnn_dropout=.0,
                 dnn_act="relu",
                 loss_mode='batch',
                 dnn_device="cuda",
                 embedding_device="cuda",
                 sparse=None,
                 use_weight=False,
                 optimizer='adagrad',
                 lr=5e-4,
                 metrics_name=('auc',),
                 best_metric_name='auc',
                 user_ids_feature_name='customer_id',
                 moe_expert_num=0,
                 moe_expert_activate_num=2):
        super(MOE, self).__init__(optimizer, lr, metrics_name, best_metric_name, user_ids_feature_name)
        self.embedding_dim = embedding_dim
        self.loss_mode = loss_mode
        self.use_weight = use_weight
        self.dnn_device = dnn_device
        self.embedding_device = embedding_device
        self.deep_emb_features = None
        self.deep_dense_features = None
        self.deep_features = None
        self.wide_features = None
        self.use_moe = False
        self.focal_loss = FocalLoss(alpha=0.95, gamma=2.0)

        # 1. 为每个 sparse 特征创建一个 Embedding 子模块
        self.sparse_feats = [f['name'] for f in feat_config['sparse']]
        self.dense_feats = [f['name'] for f in feat_config['dense']]
        self.l2_reg_embedding = l2_reg_embedding
        self.embeddings = nn.ModuleDict({
            feat['name']: self.EmbeddingModule(
                feat['vocab_size'],
                embedding_dim,
                l2_reg_embedding=l2_reg_embedding,
                init_std=embedding_init_std,
                device=embedding_device)
            for feat in feat_config['sparse']
        })

        # 2. Group dense features by normalization type for batched processing
        # Build mapping: norm_type -> list of (feature_name, feature_index)
        self.norm_type_to_features = {}
        self.dense_feat_to_index = {}  # feature_name -> index in dense_feats list
        
        for idx, feat in enumerate(feat_config['dense']):
            name = feat['name']
            norm_type = feat['norm'] if feat['norm'] is not None else 'null'
            self.dense_feat_to_index[name] = idx
            
            if norm_type not in self.norm_type_to_features:
                self.norm_type_to_features[norm_type] = []
            self.norm_type_to_features[norm_type].append(name)
        
        # Create batched normalizers for each norm type
        self.batched_normalizers = nn.ModuleDict()
        for norm_type, feat_names in self.norm_type_to_features.items():
            self.batched_normalizers[norm_type] = BatchedNormalizer(
                norm_type=norm_type,
                num_features=len(feat_names),
                device=self.dnn_device
            )

        input_dim = len(self.sparse_feats) * embedding_dim + len(feat_config['dense'])

        self.base_wdl = BaseWDL(layer_dim,
                                    deep_field_size=input_dim,
                                    wide_field_size=input_dim,
                                    l2_reg_dnn=l2_reg_dnn,
                                    l2_reg_linear=l2_reg_linear,
                                    dnn_dropout=dnn_dropout,
                                    dnn_act=dnn_act,
                                    device=dnn_device)
        if moe_expert_num > 0:
            self.use_moe = True
            self.moe_layer = MoeLayer(
                experts=[BaseWDL(layer_dim,
                                    deep_field_size=input_dim,
                                    wide_field_size=input_dim,
                                    l2_reg_dnn=l2_reg_dnn,
                                    l2_reg_linear=l2_reg_linear,
                                    dnn_dropout=dnn_dropout,
                                    dnn_act=dnn_act,
                                    device=dnn_device)
                         for _ in range(moe_expert_num)],
                gate=nn.Linear(input_dim, moe_expert_num, bias=False),
                expert_num=moe_expert_num,
                activate_num=moe_expert_activate_num,
                output_dim=1
            )
        self.to(self.dnn_device)

    def compute_predictions(self, *data, **kwargs):
        inputs = data[0]
        # sparse
        embs = []
        for name in self.sparse_feats:
            if name in inputs:
                tensor = inputs[name]
                embs.append(self.embeddings[name](tensor.to(self.embedding_device)))
        
        # dense - batched normalization
        # Prepare output tensor to hold all normalized dense features in correct order
        batch_size = next(iter(inputs.values())).shape[0]
        num_dense = len(self.dense_feats)
        dense_features_normalized = torch.zeros(batch_size, num_dense, device=self.dnn_device)
        
        # Process features grouped by normalization type
        for norm_type, feat_names in self.norm_type_to_features.items():
            # Gather all features for this norm type
            feat_tensors = []
            feat_indices = []
            for feat_name in feat_names:
                if feat_name in inputs:
                    feat_tensors.append(inputs[feat_name].to(self.dnn_device))
                    feat_indices.append(self.dense_feat_to_index[feat_name])
            
            if len(feat_tensors) > 0:
                # Stack features: (batch_size, num_features_of_this_type)
                stacked_features = torch.stack(feat_tensors, dim=1)
                
                # Apply batched normalization
                normalized = self.batched_normalizers[norm_type](stacked_features)
                
                # Place normalized features back to their correct positions
                for i, feat_idx in enumerate(feat_indices):
                    dense_features_normalized[:, feat_idx] = normalized[:, i]
        
        self.deep_emb_features = torch.cat(embs, dim=-1).to(self.dnn_device) if embs else torch.empty(batch_size, 0, device=self.dnn_device)
        self.deep_dense_features = dense_features_normalized

        input_features = torch.cat([self.deep_emb_features, self.deep_dense_features], dim=-1)

        self.input_features = input_features.to(self.dnn_device)
        if self.input_features.nelement() == 0:
            return torch.FloatTensor([])
        #增加MoeLayer
        if self.use_moe:
            predictions = self.moe_layer(self.input_features)
        else:
            predictions = self.base_wdl(self.input_features, self.input_features)
        return predictions

    # TODO add this when export onnx
#     def forward(self, *data, **kwargs):
#         deep_sparse_feat, deep_dense_feat, wide_feat = data
#         return self.compute_predictions(deep_sparse_feat, deep_dense_feat, wide_feat)

    def compute_loss(self, inputs, predictions):
        labels, weights = self._get_labels_and_weights(inputs, predictions.device, self.use_weight)
        #loss = F.binary_cross_entropy(predictions, labels, weight=weights, reduction='mean')
        loss = self.focal_loss(predictions, labels, weights)
        if self.use_moe:
            reg_loss = self.moe_layer.reg_loss()
        else:
            reg_loss = self.base_wdl.reg_loss()

        if self.loss_mode == 'batch':
            # reg_loss += (self.embedding_linear.l2_reg * torch.sum(torch.square(self.emb_linear)) +
            #              self.embedding_dnn.l2_reg * torch.sum(torch.square(self.emb_features)))
            reg_loss += self.l2_reg_embedding * torch.sum(torch.square(self.deep_emb_features))
        elif self.loss_mode == 'full':
            # reg_loss += (self.embedding_linear.reg_loss() + self.embedding_dnn.reg_loss())
            for feat in self.sparse_feats:
                reg_loss += self.embeddings[feat].reg_loss().to(self.dnn_device)
        else:
            raise ValueError('unknown loss_mode: %s' % self.loss_mode)

        total_loss = loss + reg_loss
        return total_loss

    def freeze(self, layer_dim):
        num_layers = len(layer_dim)
        for i in range(num_layers - 1):
            for p in self.base_wdl.dnn.linears[i].parameters():
                p.requires_grad = False
        return p
