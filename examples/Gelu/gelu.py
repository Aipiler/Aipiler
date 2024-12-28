import torch
import torch.nn as nn
import os


class GeluModel(nn.Module):
    def __init__(self):
        super(GeluModel, self).__init__()
        
    def forward(self, x: torch.Tensor):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def calc_model(model):
    input  = torch.tensor([[1,2,3], [3,2,1], [4,5,6], [6,5,4]], dtype=torch.float32)
    print("output:", model(input))

if __name__ == "__main__":
    model = GeluModel()
    # 创建模型实例
    model.eval()
    calc_model(model)