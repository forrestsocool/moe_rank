import time

import yaml
import torch
import glob
import os
from train_v7 import build_moe_model,feature_transform
from train_v7 import base_path, test_data_path
from datasets.OpParquetDataset import create_optimized_dataloader

# 指定权重目录
weight_path = "/data/result/MOE_train/temp/2025-08-13/epoch91_rank0.pth"
onnx_save_path = "/data/result/MOE_train/temp/2025-08-13/onnx/"


# Override forward to correctly forward args to compute_predictions
def forward_override(self, *args, **kwargs):
    # 直接把传入的 args/kwargs 透传给 compute_predictions
    return self.compute_predictions(*args, **kwargs)

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
                                                 batch_size=1024,
                                                 cache_dir="/tmp/sample_cache",
                                                 prefetch_batches=8,
                                                 num_workers=8,
                                                 use_iterable=True,
                                                 transform=feature_transform)

    dummy_inputs = {}
    # Get the first batch as dummy_inputs
    for iter_step, data in enumerate(testDataLoader):
        if iter_step == 0:
            dummy_inputs = data  # Assuming data is the dict of feature tensors
            for f in feat_cfg['labels']:
                dummy_inputs.pop(f['name'], None)
            break
    input_names = list(dummy_inputs.keys())
    # Create dynamic axes dict (avoiding | operator)
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
    #dynamic_axes.update({'output': {0: 'batch_size'}})

    # 加载权重
    state_dict = torch.load(weight_path, map_location='cuda')
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.to('cpu')
        model.eval()
        # Temporarily bind the override (or modify the class/source if possible)
        model.forward = forward_override.__get__(model)

        print(f"{'=' * 50}\n")
        if not os.path.exists(onnx_save_path):
            print("mkdir {}".format(onnx_save_path), flush=True)
            os.mkdir(onnx_save_path)

        input_names = list(dummy_inputs.keys())
        # Export to ONNX
        # The exporter will use dict keys as input names
        # torch.onnx.export(
        #     model,  # The model instance
        #     dummy_inputs,  # Dict of dummy tensors (passed as single arg)
        #     onnx_save_path + f"moe_model_{time.time()}.onnx",  # Output file path
        #     export_params=True,  # Export trained weights
        #     opset_version=15,  # Recommended; higher if needed for features like top-k in MOE
        #     do_constant_folding=True,  # Optimize by folding constants
        #     input_names=input_names,  # Use feat names as ONNX input names
        #     output_names=['output'],  # Name for the sigmoid output
        #     dynamic_axes=dynamic_axes,
        #     verbose=True  # Optional: Print export details for debugging
        # )
        torch.onnx.export(
            model,
            (dummy_inputs,),
            onnx_save_path + f"moe_model_{time.time()}.onnx",  # Output file path
            export_params=True,  # Export trained weights
            opset_version=15,
            dynamic_shapes=dynamic_axes,
            do_constant_folding=True,  # Optimize by folding constants
            input_names=["input"],
            output_names=["output"],
            dynamo=True,  # 新方式
        )





