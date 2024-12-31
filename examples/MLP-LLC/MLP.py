import torch
from torch import nn
import torch.nn.functional as F
import os


def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    # out = F.dropout(x, p=prob, training=training)
    out = residual + x
    return out


class TelechatMLP(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        hidden_size = 16
        self.gate_proj = nn.Linear(hidden_size, 10000000, bias=False, dtype=torch.float)
        self.up_proj = nn.Linear(hidden_size, 10000000, bias=False, dtype=torch.float)
        self.down_proj = nn.Linear(10000000, hidden_size, bias=True, dtype=torch.float)
        self.hidden_dropout = 0.0

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        intermediate_output = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        output = dropout_add(
            intermediate_output, residual, self.hidden_dropout, self.training
        )
        return output


def dump_model_bin(model: TelechatMLP):
    # 随机初始化权重
    model.gate_proj.weight.data.normal_(0, 0.01)
    model.up_proj.weight.data.normal_(0, 0.01)
    model.down_proj.weight.data.normal_(0, 0.01)
    model.down_proj.bias.data.normal_(0, 0.01)
    torch.save(model.state_dict(), "mlp_model.bin")


def calc_model(model: TelechatMLP):
    if not os.path.exists("mlp_model.bin"):
        dump_model_bin(model)
    model.load_state_dict(torch.load("mlp_model.bin"))
    input1 = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.float32,
    )
    input2 = torch.tensor(
        [
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        dtype=torch.float32,
    )
    print("gate_proj.weight: ", model.gate_proj.weight)
    print("up_proj.weight: ", model.up_proj.weight)
    print("down_proj.weight: ", model.down_proj.weight)
    print("down_proj.bias: ", model.down_proj.bias)
    print("output:", model(input1, input2))


if __name__ == "__main__":
    model = TelechatMLP()
    # 创建模型实例
    model.eval()
    calc_model(model)
