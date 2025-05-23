import Aipiler
import torch

def mm(a, b):
    return torch.matmul(a, b)


opt = torch.compile(mm, backend="einsum")
A = torch.randn([2, 2])
B = torch.randn([2, 2])
C = opt(A, B)