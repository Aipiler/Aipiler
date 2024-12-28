import torch
import torch.nn as nn
import os


class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, dtype=torch.float32)
        
    def forward(self, x):
        return self.embedding(x)

def dump_model_bin(model: EmbeddingModel):
    # 随机初始化权重
    model.embedding.weight.data.normal_(0, 1)
    torch.save(model.state_dict(), 'embedding_model.bin')

def calc_model(model: EmbeddingModel):
    if not os.path.exists('embedding_model.bin'):
        dump_model_bin(model)
    model.load_state_dict(torch.load('embedding_model.bin'))
    input = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 4, 6, 8, 10], [12, 14, 16, 18, 19]], dtype=torch.int)
    print("weight: ", model.embedding.weight)
    print("output:", model(input))

if __name__ == "__main__":
    model = EmbeddingModel(num_embeddings=20, embedding_dim=5)
    # 创建模型实例
    model.eval()
    calc_model(model)
    