from .utils import load_model_weights
from torch._dynamo import register_backend
from .dynamo_backend import aipiler_backend


register_backend(aipiler_backend, name="einsum")