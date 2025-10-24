import multiprocessing
import os

import numpy as np
import torch
import torch.nn
import unittest
from torch import distributed

from .embedding import Embedding, sum_pooling, mean_pooling, max_pooling
from moxing.framework.unittest.base import SingleProcessingTestCase, MultiProcessingTestCase


class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.p = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)

    def forward(self, x):
        loss = torch.sum(torch.square(x), dim=(1, 2)) / 2.0
        loss = torch.mean(loss)
        return loss



class TestEmbedding(unittest.TestCase):

    def setUp(self):
        # ids  [Batch_size, field_size_origin]
        self.ids = torch.LongTensor([[29, 69, 19, 84, 63, 78, 23, 85, 5, 5],
                                     [31, 33, 1, 30, 2, 14, 2, 41, 54, 58]])
        # wts  [Batch_size, field_size_origin]
        self.wts = torch.FloatTensor([[0.05, 0.08, 0.74, 0.02, 0.86, 0.39, 0.19, 0.81, 0.09, 0.4],
                                      [0.29, 0.06, 0.68, 0.54, 0.19, 0.69, 0.0, 0.52, 0.21, 0.22]])

        # self.wts =  torch.FloatTensor([[1.0,1.0]])

        self.pool_indices = np.array([1, 1, 2, 3, 1, 2])

        self.emb = Embedding(
            num_embeddings=100,
            embedding_dim=4,
            l2_reg_embedding=0.)

    def test_sum_pooling_embedding(self):
        emb_vector = self.emb(self.ids, self.wts)
        assert emb_vector.shape == (2, 10, 4)

        # origin embedding vector shape is [batch_size, filed_size_origin, embedding_dim]
        # sum embedding vector shape is [batch_size, filed_size_pool, embedding_dim]
        # filed_size_pool = len(pool_indices)
        sum_emb_vector = sum_pooling(emb_vector, self.pool_indices)
        assert sum_emb_vector.shape == (2, len(self.pool_indices), 4)

        # origin embedding vector (column 1, 2, 8)(in dim=1) should be equal to
        # sum embedding vector (column 1, 2, 5)(in dim=1).
        assert torch.equal(emb_vector[:, [0, 1, 7], :], sum_emb_vector[:, [0, 1, 4], :])

        # do sum in origin embedding vector (column 3, 4)(in dim=1) should be equal to
        # sum embedding vector (column 3)(in dim=1).
        assert torch.equal(torch.sum(emb_vector[:, [2, 3], :], dim=1, keepdim=True), sum_emb_vector[:, [2], :])

        # do sum in origin embedding vector (column 5, 6, 7)(in dim=1) should be equal to
        # sum embedding vector (column 4)(in dim=1).
        assert torch.equal(torch.sum(emb_vector[:, [4, 5, 6], :], dim=1, keepdim=True), sum_emb_vector[:, [3], :])

        # do sum in origin embedding vector (column 9, 10)(in dim=1) should be equal to
        # sum embedding vector (column 6)(in dim=1).
        assert torch.equal(torch.sum(emb_vector[:, [8, 9], :], dim=1, keepdim=True), sum_emb_vector[:, [5], :])

        print("sum pooling test success!")

    def test_mean_pooling_embedding(self):
        emb_vector = self.emb(self.ids, self.wts)
        assert emb_vector.shape == (2, 10, 4)

        # origin embedding vector shape is [batch_size, filed_size_origin, embedding_dim]
        # mean embedding vector shape is [batch_size, filed_size_pool, embedding_dim]
        # filed_size_pool = len(pool_indices)
        mean_emb_vector = mean_pooling(emb_vector, self.pool_indices)
        assert mean_emb_vector.shape == (2, len(self.pool_indices), 4)

        # origin embedding vector (column 1, 2, 8)(in dim=1) should be equal to
        # mean embedding vector (column 1, 2, 5)(in dim=1).
        assert torch.equal(emb_vector[:, [0, 1, 7], :], mean_emb_vector[:, [0, 1, 4], :])

        # do mean in origin embedding vector (column 3, 4)(in dim=1) should be equal to
        # mean embedding vector (column 3)(in dim=1).
        assert torch.equal(torch.mean(emb_vector[:, [2, 3], :], dim=1, keepdim=True), mean_emb_vector[:, [2], :])

        # do mean in origin embedding vector (column 5, 6, 7)(in dim=1) should be equal to
        # mean embedding vector (column 4)(in dim=1).
        assert torch.equal(torch.mean(emb_vector[:, [4, 5, 6], :], dim=1, keepdim=True), mean_emb_vector[:, [3], :])

        # do mean in origin embedding vector (column 9, 10)(in dim=1) should be equal to
        # mean embedding vector (column 6)(in dim=1).
        assert torch.equal(torch.mean(emb_vector[:, [8, 9], :], dim=1, keepdim=True), mean_emb_vector[:, [5], :])

        print("mean pooling test success!")

    def test_max_pooling_embedding(self):
        emb_vector = self.emb(self.ids, self.wts)
        assert emb_vector.shape == (2, 10, 4)

        # origin embedding vector shape is [batch_size, filed_size_origin, embedding_dim]
        # max embedding vector shape is [batch_size, filed_size_pool, embedding_dim]
        # filed_size_pool = len(pool_indices)
        max_emb_vector = max_pooling(emb_vector, self.pool_indices)
        assert max_emb_vector.shape == (2, len(self.pool_indices), 4)

        # origin embedding vector (column 1, 2, 8)(in dim=1) should be equal to
        # max embedding vector (column 1, 2, 5)(in dim=1).
        assert torch.equal(emb_vector[:, [0, 1, 7], :], max_emb_vector[:, [0, 1, 4], :])

        # do mean in origin embedding vector (column 3, 4)(in dim=1) should be equal to
        # max embedding vector (column 3)(in dim=1).
        assert torch.equal(torch.max(emb_vector[:, [2, 3], :], dim=1, keepdim=True)[0], max_emb_vector[:, [2], :])

        # do mean in origin embedding vector (column 5, 6, 7)(in dim=1) should be equal to
        # max embedding vector (column 4)(in dim=1).
        assert torch.equal(torch.max(emb_vector[:, [4, 5, 6], :], dim=1, keepdim=True)[0], max_emb_vector[:, [3], :])

        # do mean in origin embedding vector (column 9, 10)(in dim=1) should be equal to
        # max embedding vector (column 6)(in dim=1).
        assert torch.equal(torch.max(emb_vector[:, [8, 9], :], dim=1, keepdim=True)[0], max_emb_vector[:, [5], :])

        print("max pooling test success!")

