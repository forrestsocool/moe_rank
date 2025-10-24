import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, expert_num: int, activate_num: int, output_dim: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.expert_num = expert_num
        self.activate_num = activate_num
        self.output_dim = output_dim
        #增加dropout防止极化
        self.dropout = nn.Dropout(p=1.0/expert_num)  # 假设dropout率为0.5
    # def forward(self, deep_inputs: torch.Tensor, wide_inputs: torch.Tensor):
    #     gate_input = torch.cat([deep_inputs, wide_inputs], dim=1)
    #     gate_logits = self.gate(gate_input)
    #     weights, selected_experts = torch.topk(gate_logits, self.activate_num)
    #     weights = F.softmax(weights, dim=1, dtype=torch.float).to(gate_input.dtype)
    #     bz = deep_inputs.shape[0]
    #     results = torch.zeros(bz, self.output_dim).to(weights.device)
    #     for i, expert in enumerate(self.experts):
    #         batch_idx, nth_expert = torch.where(selected_experts == i)
    #         results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
    #             deep_inputs[batch_idx],
    #             wide_inputs[batch_idx]
    #         )
    #     return results

    def forward(self, inputs: torch.Tensor):
        gate_input = torch.cat([inputs], dim=1)
        gate_logits = self.gate(gate_input)
        gate_logits = self.dropout(gate_logits)
        weights, selected_experts = torch.topk(gate_logits, self.activate_num)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(gate_input.dtype)
        bz = inputs.shape[0]
        results = torch.zeros(bz, self.output_dim, device=weights.device)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            #print(f'{i} : {batch_idx.shape}')
            expert_output = expert(
                inputs.index_select(0, batch_idx),
                inputs.index_select(0, batch_idx)
            )
            # Multiply weights with corresponding expert outputs
            weighted_output = expert_output * weights.index_select(0, batch_idx).gather(1, nth_expert.unsqueeze(1))
            # Sum up the weighted expert outputs using index_add_
            results.index_add_(0, batch_idx, weighted_output)
        return results

    #全部计算
    # def forward(self, deep_inputs: torch.Tensor, wide_inputs: torch.Tensor):
    #     gate_input = torch.cat([deep_inputs, wide_inputs], dim=1)
    #     gate_logits = self.gate(gate_input)
    #     weights, selected_experts = torch.topk(gate_logits, self.activate_num)
    #     weights = F.softmax(weights, dim=1).view(-1, self.activate_num, 1).to(gate_input.dtype)
    #     bz = deep_inputs.shape[0]
    #
    #     expert_outputs = torch.stack([expert(deep_inputs, wide_inputs) for expert in self.experts], dim=1)
    #     # Using advanced indexing to select the desired expert outputs based on `selected_experts`
    #     selected_expert_outputs = expert_outputs[torch.arange(bz).unsqueeze(1), selected_experts]
    #
    #     # Perform element-wise multiplication of weights with the selected expert outputs
    #     weighted_expert_outputs = weights * selected_expert_outputs
    #
    #     # Sum up the contributions from each expert for each data point
    #     results = weighted_expert_outputs.sum(dim=1)
    #
    #     return results


    def reg_loss(self):
        reg_losses = []
        # 遍历所有的experts
        for expert in self.experts:
            # 调用expert的reg_loss方法并添加到列表中
            reg_losses.append(expert.reg_loss())
        # 使用torch.stack()将列表中的张量堆叠成一个张量
        reg_losses_tensor = torch.stack(reg_losses)
        # 使用torch.mean()计算张量的均值
        mean_reg_loss = torch.mean(reg_losses_tensor)
        return mean_reg_loss