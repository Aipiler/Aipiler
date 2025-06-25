import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from config import Config, batch_size, seq_len


# Retrieve from the website https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py
class DeepseekV3MLP(nn.Module):
    def __init__(self, config: Config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


config = Config()
model = DeepseekV3MLP(config)

input_tensor = torch.randn(
    [batch_size, seq_len, config.hidden_size], dtype=torch.float32
)
output_tensor = model(input_tensor)

print(output_tensor)
