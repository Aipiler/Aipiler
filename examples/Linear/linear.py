import torch

import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = LinearModel(input_size=10, output_size=5)
model.eval()
# 从当前文件夹下的linear_model.bin中加载model的参数
# model.load_state_dict(torch.load('linear_model.bin'))

# 随机初始化权重
model.linear.weight.data.normal_(0, 0.01)
model.linear.bias.data.zero_()

# 调用model的forward函数
# input_data = torch.randn(2, 10)  # 创建输入数据
# print(input_data.dtype)
# output = model(input_data)  # 调用forward函数
# print(output)


# 导出模型参数为bin文件
torch.save(model.state_dict(), 'linear_model.bin')