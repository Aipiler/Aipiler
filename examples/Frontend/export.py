import torch
import Aipiler

class MyModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y

ep = torch.export.export(MyModule(), (torch.randn(5, 5), torch.randn(5, 5)))

g = Aipiler.compile(ep)
