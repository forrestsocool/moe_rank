import yaml
import torch
import glob
import os
from train_v7 import build_moe_model,feature_transform
from train_v7 import base_path, test_data_path
from datasets.OpParquetDataset import create_optimized_dataloader

# 指定权重目录
weight_dir = "/data/result/MOE_train/temp/2025-08-13/"

if __name__ == "__main__":
    def load_config(path="scripts/train_v7.yaml"):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    feat_cfg = load_config()
    layer_dim = [1024, 512, 256, 128]
    model = build_moe_model(feat_cfg = feat_cfg,
                             layer_dim=layer_dim,
                             embedding_dim=24,
                             device='cuda',
                             optimizer='adamw',
                             lr=5e-4,
                             moe_num=8,
                             moe_activate=2,
                             metrics_name=('auc', 'gauc'))

    testDataLoader, _ = create_optimized_dataloader(base_path=base_path + test_data_path,
                                                 feat_config=feat_cfg,
                                                 batch_size=10240,
                                                 cache_dir="/tmp/sample_cache",
                                                 prefetch_batches=8,
                                                 num_workers=8,
                                                 use_iterable=True,
                                                 transform=feature_transform)

    # 获取所有.pt权重文件
    weight_files = glob.glob(os.path.join(weight_dir, "*.pth"))

    # 遍历并评测每个权重文件
    for weight_path in weight_files:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {weight_path}")

        # 加载权重
        state_dict = torch.load(weight_path, map_location='cuda')
        model.load_state_dict(state_dict)

        # 设置为评估模式
        model.eval()

        # 执行评测
        with torch.no_grad():
            model.eval_epoch(testDataLoader)
        print(f"{'=' * 50}\n")

