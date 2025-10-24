import time

import yaml
import torch
from torch.utils.data import DataLoader
from OpParquetDataset import create_optimized_dataloader
import pandas as pd
# 加载特征配置
with open("./scripts/train_v7.yaml", "r") as f:
    feat_config = yaml.safe_load(f)

def feature_transform(input: pd.DataFrame) -> pd.DataFrame:
    # 确保 'label' 列存在
    if 'label' in input.columns:
        # 对 'label' 列进行处理：大于2为1，否则为0
        input['label'] = input['label'].apply(lambda x: 1 if x > 2 else 0)
    else:
        raise ValueError("no label found")

    return input

# 构建数据集
# dataset = MyParquetDataset(
#     base_path="/data/ctr_samples/dt=2025-07-24",  # 替换为你自己的 parquet 文件路径
#     feat_config=feat_config,
#     #limit=100  # 读取前100条
# )

# 构建DataLoader
# dataloader = DataLoader(dataset, batch_size=10240, shuffle=False, collate_fn=lambda x: x)


my_dataloader, dataset = create_optimized_dataloader(base_path="/data/ctr_samples/dt=2025-07-24",
                                            feat_config=feat_config,
                                            batch_size=10240,
                                            cache_dir="/tmp/sample_cache",
                                            use_iterable=True,
                                            transform=feature_transform)
#dataset.cleanup()

# print(dataset.__len__())
# 读取一个 batch 进行测试
# for batch in my_dataloader:
#     # batch 是长度为 batch_size 的 list，每个元素是 (name, tensor) 的 list
#     print(f"Batch size: {len(batch)}")
#     for i, sample in enumerate(batch):
#         print(f"Sample {i}:")
#         for name, tensor in sample:
#             print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, value={tensor}")
#     break  # 只测试第一个 batch

total_len = 0
for i, batch in enumerate(my_dataloader):
    print(f"Batch size: {len(batch['is_sku_added'])}")
    total_len += len(batch['is_sku_added'])
    time.sleep(0.2)
    # sample = batch[0]  # 取第一个样本
    # print(f"Sample structure: {sample}")  # 查看样本的实际结构
    # print(f"Sample type: {type(sample)}")
    # print(f"Sample length: {len(sample)}") if hasattr(sample, '__len__') else None
    #print(batch)
    #break
print(total_len)


# benchmark_dataloader(my_dataloader, num_batches=10)