import torch
from torch import nn
import os


class MixedFusedRMSNorm(nn.Module):
    # Extracted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = MixedFusedRMSNorm()

    def forward(self, hidden_states: torch.Tensor):
        return self.rms(hidden_states)


def dump_model_bin(model: TestModule):
    # 随机初始化权重
    model.rms.weight.data.normal_(0, 0.01)
    torch.save(model.state_dict(), "rms_model.bin")


def calc_model(model: TestModule):
    # if not os.path.exists("rms_model.bin"):
    dump_model_bin(model)
    model.load_state_dict(torch.load("rms_model.bin"))
    input = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.float16,
    )
    print("weight: ", model.rms.weight)
    print("output:", model(input))


if __name__ == "__main__":
    model = TestModule()
    # 创建模型实例
    model.eval()
    calc_model(model)
