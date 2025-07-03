from Aipiler import dsl
from Aipiler import datatype as dtypes
from Aipiler.tensor import *
from Aipiler.dim import dims
import iree.runtime as rt
import torch


class Module:
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
    def forward(self, *args, **kwds):
        raise NotImplementedError()

class Matmul(Module):
    def __init__(self, w: Parameter):
        self.w = w
        
    def forward(self, x: FakeTensor):
        t = dsl.map(x, self.w, "ik, kj -> ikj", "*")
        y = dsl.reduce(t, "ikj -> ij", "k", "+")
        return y
    
class _3MM(Module):
    def __init__(self, weights: List[Parameter]):
        super().__init__()
        self.layers = [Matmul(w) for w in weights]
    
    def forward(self, x: FakeTensor):
        # return matmul(matmul(matmul(x, w1), w2), w3)
        for model in self.layers:
            x = model(x)
        return x
        

_weights = [
    torch.randn([2, 2], dtype=torch.float32),
    torch.randn([2, 2], dtype=torch.float32),
    torch.randn([2, 2], dtype=torch.float32)
]

weights = [
    from_torch_to_parameter(_w) for _w in _weights
]
model = _3MM(weights)

x = FakeTensor(dims("a", "b"), dtypes.float32)
example_inputs = [x]



compiled_binary = dsl.compile_module(model, example_inputs, target_backend="host")

config = rt.Config("local-task")
vm_module = rt.VmModule.copy_buffer(
    config.vm_instance, compiled_binary.map_memory()
)

np_x = np.random.rand(2, 2).astype(np.float32)
inputs = [np_x]

vmm = rt.load_vm_module(
    vm_module,
    config,
)

y = vmm.main(*inputs)
print("aipiler result: \n", y.to_host())