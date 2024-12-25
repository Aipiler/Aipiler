# load_model.py
import torch

def load_model_weights(model_path):
    # 加载模型的权重
    model_weights = torch.load(model_path)
    
    # 提取并返回模型的所有权重
    weights_dict = {}
    for key, value in model_weights.items():
        weights_dict[key] = value.cpu().numpy().tolist()  # 转换为 Python 列表
    return weights_dict
