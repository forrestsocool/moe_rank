import numpy as np
import torch.nn as nn
import torch
from torch import distributed


def get_num_ids_this_node(num_ids, world_size, rank):
    """
    return the number of id need to be calculated in this node
    :param num_ids: the total number of ids
    :param world_size: the number of processes in the current process group, / or 1 when dist is disabled
    :param rank:  a unique identifier assigned to each process within a distributed
    process group, from 0 - world_size
    :return:
    """
    max_id = num_ids - 1
    #能整除均分，不能整除最后一个process处理数目+1
    num_ids_this_node = max_id // world_size + (1 if rank <= max_id % world_size else 0)
    return num_ids_this_node


def ragged_all_gather(tensor, group=None):
    world_size = distributed.get_world_size() if distributed.is_initialized() else 1
    if world_size <= 1:
        return [tensor]

    tensor_len = tensor.shape[0]
    tensor_len_list = [torch.zeros(size=[1], dtype=torch.int32, device=tensor.device)
                       for _ in range(world_size)]
    distributed.all_gather(tensor_len_list,
                           torch.IntTensor([tensor_len]).to(tensor.device), group=group)
    max_tensor_len = torch.max(torch.cat(tensor_len_list)).detach().cpu().numpy()

    padding_len = max_tensor_len - tensor_len
    if padding_len > 0:
        tensor_pad = torch.zeros(size=[padding_len], dtype=torch.int64, device=tensor.device)
        tensor = torch.cat([tensor, tensor_pad])

    tensor_list = [torch.zeros(size=[max_tensor_len], dtype=tensor.dtype, device=tensor.device)
                   for _ in range(world_size)]
    distributed.all_gather(tensor_list, tensor, group=group)

    gathered_tensor = []
    for i, tensor_each_node in enumerate(tensor_list):
        tensor_each_node = tensor_each_node[:tensor_len_list[i]].to(tensor.device)
        gathered_tensor.append(tensor_each_node)

    return gathered_tensor


def reduce_unique(tensor, return_inverse=False, group=None):
    unique_id = torch.unique(tensor, sorted=False)
    global_unique_ids = ragged_all_gather(unique_id, group=group)
    global_unique_ids = torch.cat(global_unique_ids)

    # get inverse with given unique values
    if return_inverse:
        ret = torch.unique(global_unique_ids, sorted=True)
        # 按索引存储
        ret = (ret, torch.bucketize(tensor, ret))
    else:
        ret = torch.unique(global_unique_ids, sorted=False)

    return ret


def sum_pooling(emb_vector, pool_indices):
    """
    emb_vector 按 pool_indices 邻近式求和降维，降维后embedding的位数不变
    :param emb_vector: 输入待降维embbeding
    :param pool_indices: 1维，长度为sum_pooling后返回的emb_vector维度, 每个位置的元素表示与周边几个向量临近求和
    :return:
    """
    sizes = torch.tensor(pool_indices, dtype=torch.long)
    # prepare an index vector for summation
    ind = torch.arange(len(sizes)).repeat_interleave(sizes).to(emb_vector.device)
    # prepare the output
    sum_pooled_vector = torch.zeros([emb_vector.shape[0], len(sizes), emb_vector.shape[2]]).to(emb_vector.device)
    # do the actual summation
    sum_pooled_vector.index_add_(1, ind, emb_vector)
    return sum_pooled_vector


def mean_pooling(emb_vector, pool_indices):
    """
    获得sum_pooling求和的结果后，每一行取求和个数的平均
    :param emb_vector: 输入待降维embbeding
    :param pool_indices:  1维，长度为sum_pooling后返回的emb_vector维度, 每个位置的元素表示与周边几个向量临近求和
    :return:
    """
    # do the sum pooling
    sum_pooled_vector = sum_pooling(emb_vector, pool_indices)
    sizes = torch.tensor(pool_indices, dtype=torch.int32).reshape(1, -1, 1).to(emb_vector.device)
    # do the mean
    mean_pooled_vector = sum_pooled_vector / sizes
    return mean_pooled_vector


def max_pooling(emb_vector, pool_indices):
    # TODO (This can be speed-up when torch support scatter_reduce)
    tmp_vector_list = list(torch.split(emb_vector, list(map(int, pool_indices)), dim=1))
    for i, tmp_vector in enumerate(tmp_vector_list):
        tmp_vector_list[i] = torch.max(tmp_vector, dim=1)[0]
    max_pooled_vector = torch.cat(tmp_vector_list, dim=1).reshape(emb_vector.shape[0], -1, emb_vector.shape[2])
    return max_pooled_vector


class BaseEmbedding(nn.Module):
    def __init__(self,
                 l2_reg_embedding=1e-5,
                 pool_mode=None,
                 pool_indices=None):
        super(BaseEmbedding, self).__init__()
        self.l2_reg = torch.FloatTensor(np.array(l2_reg_embedding))
        self.pool_mode = pool_mode
        self.pool_indices = pool_indices

    def feature_pool(self, emb_vector):
        # Pooling for sequence feature
        # ids: [batch_size, field_size_o]; wts: [batch_size, field_size_o]
        # filed_size_o = v_num + c_num + sum(mc_len)
        # filed_size_d = v_num + c_num + sum(pool_len)
        # emb_vector: [batch_size, filed_size_o, embedding_dim]
        #                    ||
        #                    ||
        #                    \/
        # pooled_vector: [batch_size, filed_size_d, embedding_dim]
        if self.pool_mode == 'sum':
            pooled_vector = sum_pooling(emb_vector, self.pool_indices)
        elif self.pool_mode == 'max':
            pooled_vector = max_pooling(emb_vector, self.pool_indices)
        elif self.pool_mode == 'mean':
            pooled_vector = mean_pooling(emb_vector, self.pool_indices)
        else:
            raise ValueError("pool mode has to be one of sum, mean or max")

        return pooled_vector

    def reg_loss(self):
        return self.l2_reg * torch.sum(torch.square(self.emb.weight))

class NoWtsEmbedding(BaseEmbedding):
    """
    无特征权重embedding
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 l2_reg_embedding=1e-5,
                 init_std=0.0001,
                 sparse=None,
                 pool_mode=None,
                 pool_indices=None,
                 device='cpu'):
        """
        :param num_embeddings (int): size of the dictionary of embeddings
        :param embedding_dim (int): the size of each embedding vector
        :param l2_reg_embedding:
        :param init_std:
        :param sparse:
        :param pool_mode:
        :param pool_indices:
        :param device:
        """
        super(NoWtsEmbedding, self).__init__(l2_reg_embedding=l2_reg_embedding, pool_mode=pool_mode,
                                        pool_indices=pool_indices)
        if sparse is None:
            sparse = (device == 'cpu')
        self.emb = nn.Embedding(num_embeddings, embedding_dim, sparse=sparse, scale_grad_by_freq=True)
        nn.init.normal_(self.emb.weight, mean=0, std=init_std)
        self.device = device
        self.to(device)

    def forward(self, ids):
        ids = ids.to(self.device) % self.emb.num_embeddings
        emb_vector = self.emb(ids)
        if self.pool_mode:
            emb_vector = self.feature_pool(emb_vector)
        else:
            emb_vector = emb_vector
        return emb_vector


class Embedding(BaseEmbedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 l2_reg_embedding=1e-5,
                 init_std=0.0001,
                 sparse=None,
                 pool_mode=None,
                 pool_indices=None,
                 device='cpu'):
        """
        :param num_embeddings (int): size of the dictionary of embeddings
        :param embedding_dim (int): the size of each embedding vector
        :param l2_reg_embedding:
        :param init_std:
        :param sparse:
        :param pool_mode:
        :param pool_indices:
        :param device:
        """
        super(Embedding, self).__init__(l2_reg_embedding=l2_reg_embedding, pool_mode=pool_mode,
                                        pool_indices=pool_indices)
        if sparse is None:
            sparse = (device == 'cpu')
        self.emb = nn.Embedding(num_embeddings, embedding_dim, sparse=sparse)
        nn.init.normal_(self.emb.weight, mean=0, std=init_std)
        self.device = device
        self.to(device)

    def forward(self, ids, wts):
        ids = ids.to(self.device)
        wts = wts.to(self.device)
        emb_vector = self.emb(ids)
        wts = torch.unsqueeze(wts, -1)

        if self.pool_mode:
            emb_vector = self.feature_pool(emb_vector)
            # TODO(jqf) this should be rewrite when features is out-of-order(values features after category features)
            emb_vector = emb_vector * wts[:, :len(self.pool_indices)]
        else:
            emb_vector = emb_vector * wts

        return emb_vector