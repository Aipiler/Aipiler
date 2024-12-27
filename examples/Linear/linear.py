import torch
import torch.nn as nn
import os


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

def dump_model_bin(model):
    # 随机初始化权重
    model.linear.weight.data.normal_(0, 0.01)
    model.linear.bias.data.normal_(0, 0.01)
    torch.save(model.state_dict(), 'linear_model.bin')

def calc_model(model):
    if not os.path.exists('linear_model.bin'):
        dump_model_bin(model)
    model.load_state_dict(torch.load('linear_model.bin'))
    input  = torch.tensor([[0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19]], dtype=torch.float32)
    print("weight: ", model.linear.weight)
    print("bias: ", model.linear.bias)
    print("output:", model(input))

if __name__ == "__main__":
    model = LinearModel(input_size=10, output_size=5)
    # 创建模型实例
    model.eval()
    calc_model(model)
    