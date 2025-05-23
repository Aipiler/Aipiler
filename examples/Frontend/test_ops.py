from Aipiler.ops import *
from Aipiler.tensor import Tensor
from Aipiler.dim import Dim

a = Tensor([Dim(None), Dim(None)])
b = Tensor([Dim(None), Dim(None)])
c = matmul(a, b)
d = Tensor([Dim(None), Dim(None)])
e = add(c, d)