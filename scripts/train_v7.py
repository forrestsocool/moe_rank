import sys, os
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)

import datetime
import torch
import torch.multiprocessing
from torch import distributed
#from moxing.pytorch.distributed.dist_util import launcher
from models.moe import MOE
from torch.utils.data import Dataset, DataLoader
import tempfile
from moxing.framework.file import file_io
import pandas as pd
import numpy as np
import time
import tempfile
import pickle
import argparse
from datasets.OpParquetDataset import create_optimized_dataloader
import yaml


num_epochs = 3
layer_dim = [1024, 512, 256, 128]

weight_path = '/data/result/MOE_train'
base_path = '/data/ctr_samples/'
train_data_path_list = [
'dt=2025-07-03','dt=2025-07-09','dt=2025-07-15','dt=2025-07-21','dt=2025-07-27',
'dt=2025-07-04','dt=2025-07-10','dt=2025-07-16','dt=2025-07-22','dt=2025-07-28',
'dt=2025-07-05','dt=2025-07-11','dt=2025-07-17','dt=2025-07-23','dt=2025-07-29',
'dt=2025-07-06','dt=2025-07-12','dt=2025-07-18','dt=2025-07-24','dt=2025-07-30',
'dt=2025-07-07','dt=2025-07-13','dt=2025-07-19','dt=2025-07-25','dt=2025-07-31',
'dt=2025-07-08','dt=2025-07-14','dt=2025-07-20','dt=2025-07-26','dt=2025-08-01',
'dt=2025-08-02','dt=2025-08-03','dt=2025-08-04']
test_data_path = 'dt=2025-08-05'



def load_config(path="scripts/train_v7.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
feat_cfg = load_config()


def feature_transform(input: pd.DataFrame) -> pd.DataFrame:
    # 确保 'label' 列存在
    if 'label' in input.columns:
        # 对 'label' 列进行处理：大于2为1，否则为0
        input['label'] = input['label'].apply(lambda x: 1 if x > 2 else 0)
    else:
        raise ValueError("no label found")

    return input

def build_moe_model( feat_cfg,
                     layer_dim,
                     embedding_dim=4,
                     loss_mode="batch",
                     use_weight=False,
                     device='cuda',
                     optimizer='adamw',
                     lr=0.00025,
                     moe_num=0,
                     moe_activate=2,
                     metrics_name=('auc',),
                     ):
    model = MOE( feat_config=feat_cfg,
                        embedding_dim=embedding_dim,
                        layer_dim=layer_dim,
                        loss_mode=loss_mode,
                        use_weight=use_weight,
                        dnn_dropout=0.5,
                        dnn_device=device,
                        embedding_device=device,
                        optimizer=optimizer,
                        lr = lr,
                        l2_reg_embedding=1e-2,
                        l2_reg_dnn=1e-3,
                        l2_reg_linear=1e-4,
                        moe_expert_num=moe_num,
                        moe_expert_activate_num=moe_activate,
                        metrics_name=metrics_name,
                    )
    return model

# def eval_epoch(model, testDataLoader=None):
#     import gc
#     if testDataLoader != None:
#         #single_dataset = MySingleDataset(test_data_path, deep_sparse_cols, deep_dense_cols, wide_cols)
#         model.eval_epoch(testDataLoader)
#     else:
#         single_dataset = load_data(test_data_path)
#         TestDataLoader = DataLoader(dataset=single_dataset, shuffle=False, batch_size=40960, num_workers=8)
#         model.eval_epoch(TestDataLoader)
#         del single_dataset
#         del TestDataLoader
#     gc.collect()

# def load_data(file_path, push_dict=False):
#     import logging
#     import os
#     logging.info(f"[{time.strftime('%H:%M:%S')}] preloading file : {file_path} start")
#     file = open (file_path, 'rb')
#     train_dataset = pickle.load(file)
#     file.close()
#     logging.info(f"[{time.strftime('%H:%M:%S')}] preloading file : {file_path} finish")
#     if push_dict:
#         pre_dict[file_path] = train_dataset
#     return train_dataset

def prepare_for_train(lr=5e-4, output_dir=weight_path, moe_num=8, moe_activate=2):
    is_dist = distributed.is_initialized()
    rank = distributed.get_rank() if is_dist else 0
    is_chief = (rank == 0)
    model = build_moe_model(feat_cfg = feat_cfg,
                             layer_dim=layer_dim,
                             embedding_dim=24,
                             device='cuda',
                             optimizer='adamw',
                             lr=lr,
                             moe_num=moe_num,
                             moe_activate=moe_activate)
    print(next(model.parameters()).device)
    today_str = datetime.date.today()
    temp_dir = os.path.join(output_dir, 'temp', str(today_str))
    if not os.path.exists(temp_dir):
        print("mkdir {}".format(temp_dir))
        #os.mkdir(temp_dir)
        os.makedirs(os.path.join(temp_dir,"model"))
    return is_chief, model, rank, temp_dir

#@launcher
def train():
    is_chief, model, rank, temp_dir = prepare_for_train(moe_num=8, moe_activate=2)
    testDataLoader, _ = create_optimized_dataloader(base_path=base_path + test_data_path,
                                                 feat_config=feat_cfg,
                                                 batch_size=10240,
                                                 cache_dir="/tmp/sample_cache",
                                                 prefetch_batches=8,
                                                 num_workers=8,
                                                 use_iterable=True,
                                                 transform=feature_transform)
    import gc, threading
    for epoch in range(num_epochs):
        # train method 2
        for index in range(0, len(train_data_path_list)):
            train_data_path = base_path + train_data_path_list[index]
            print(f'processing training data : {train_data_path}')
            train_dataloader, dataset = create_optimized_dataloader(base_path=train_data_path,
                                                           feat_config=feat_cfg,
                                                           batch_size=10240,
                                                           cache_dir="/tmp/sample_cache",
                                                           prefetch_batches=8,
                                                           num_workers=8,
                                                           use_iterable=True,
                                                           transform=feature_transform)
            model.train_epoch(train_dataloader)
            dataset.cleanup()
            del train_dataloader
            gc.collect()
            if index%4 == 0:
                model.eval()
                model.eval_epoch(testDataLoader)
                model.save_checkpoint(temp_dir, False, is_chief, rank)
        model.eval()
        # eval
        model.eval_epoch(testDataLoader)
        # save model
        model.save_checkpoint(temp_dir, False, is_chief, rank)
    #
    # # configure the files of modelarts inference service
    # configure_modelarts_infer(meta=train_meta, flags=FLAGS)
    return

# pre_dict = {}
# sample_saved = False

parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=str, default='2023-04-21',help="target data dt, 格式%Y-%m-%d")
parser.add_argument('--train_data_path', type=str,help="训练数据，会被modelarts下载到本地，因此需要指定分区")
parser.add_argument('--weight_path', type=str,help="预训练权重")
parser.add_argument('--output_path', type=str,help="模型保存路径")
args, unknown = parser.parse_known_args()

if __name__ == "__main__":
    import logging
    # 设置日志级别为 WARNING
    logging.basicConfig(level=logging.WARNING)

    # train_data_path = args.train_data_path
    # output_path = args.output_path
    # weight_path = args.weight_path
    torch.multiprocessing.set_sharing_strategy('file_system')
    train()