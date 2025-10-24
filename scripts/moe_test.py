import yaml
import torch
from models.moe import MOE


def load_config(path="./models/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # 1. 加载配置
    cfg = load_config()
    feat_cfg = cfg['feat_config']

    # 2. 初始化模型（使用 CPU 进行测试）
    model = MOE(
        layer_dim=[16, 8],
        embedding_dim=4,
        feat_config=feat_cfg,
        moe_expert_num=2,
        moe_expert_activate_num=2,
        dnn_device='cpu',
        embedding_device='cpu'
    )

    # 3. 构造伪造输入数据
    batch_size = 4
    inputs = []

    # Sparse features: 随机整数索引
    for feat in feat_cfg['sparse']:
        name = feat['name']
        vocab_size = feat['vocab_size']
        tensor = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)
        inputs.append((name, tensor))

    # Dense features: 随机浮点数
    for feat in feat_cfg['dense']:
        name = feat['name']
        tensor = torch.rand(batch_size)
        inputs.append((name, tensor))

    print(inputs)

    # 4. 测试前向计算
    preds = model.compute_predictions(*inputs)
    print(f"Predictions shape: {preds.shape}")
    print(preds)

    # 5. （可选）测试损失计算
    # 如果需要测试 compute_loss，需要构造 labels:
    # labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float)
    # loss = model.compute_loss((labels, None), preds)
    # print(f"Loss: {loss.item()}")

if __name__ == '__main__':
    main()