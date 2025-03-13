import torch
from torch import nn
import os


class MyModel(nn.Module):
    # Extracted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def __init__(self, hidden_size=2049):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

    def forward(self):
        return self.weight

class Test(nn.Module):
    # Extracted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def __init__(self, hidden_size=2049):
        super().__init__()
        self.test = MyModel()

    def forward(self):
        return self.test()

def dump_model_bin(model: Test):
    # 随机初始化权重
    model.test.weight.data.normal_(0, 0.01)
    torch.save(model.state_dict(), "weight_model.bin")


def calc_model(model: Test):
    # if not os.path.exists("weight_model.bin"):
    # dump_model_bin(model)
    torch.save(model.state_dict(), "weight_model.bin")
    model.load_state_dict(torch.load("weight_model.bin"))
    print("weight: ", model.test.weight)
    print("output:", model())


if __name__ == "__main__":
    model = Test()
    # 创建模型实例
    model.eval()
    calc_model(model)
