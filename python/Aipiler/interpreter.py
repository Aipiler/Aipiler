from typing import Dict, Any, Callable, Optional, Tuple, Set, List, Type
from Aipiler.registry import Registry
from Aipiler.tensor import FakeTensor, from_torch_to_fake_tensor
import logging
import inspect
import torch

logger = logging.getLogger(__name__)


def is_torch_method_inplace(name: str):
    return name[-1] == "_"


class Interpreter:
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__()
        self.graph_module: torch.fx.GraphModule = graph_module
        self.graph: torch.fx.Graph = graph_module.graph
        self.torch_modules: Dict[str, torch.nn.Module] = dict(
            graph_module.named_modules()
        )

        self._check_support()

    def __call__(self, *args):
        return self.forward(*args)

    def _check_support(self):
        not_supported = set()
        for node in self.graph.nodes:
            if node.op == "call_function":
                converted_fn: Optional[Callable] = self._lookup_aipiler_function(
                    node.target
                )
                if converted_fn is None:
                    not_supported.add(node.target)

        if len(not_supported) > 0:
            raise NotImplementedError(
                "\n".join([target.__name__ for target in not_supported])
            )

    def _lookup_aipiler_function(self, torch_func) -> Callable:
        if torch_func not in Registry.registered_functions:
            raise NotImplementedError(torch_func.__name__)
        return Registry.registered_functions[torch_func]

    @staticmethod
    def _callable_info(f: Callable) -> Tuple[str, str, int]:
        if inspect.isbuiltin(f):
            callable_name = f.__name__
            filename = "builtin"
            lineno = 0
        else:
            if inspect.ismethod(f):
                func = dict(inspect.getmembers(f))["__func__"]
                code = dict(inspect.getmembers(func))["__code__"]
                callable_name = f.__qualname__
            elif inspect.isfunction(f):
                code = dict(inspect.getmembers(f))["__code__"]
                callable_name = f.__qualname__
            else:
                # an object with __call__ method
                func = dict(inspect.getmembers(getattr(f, "__call__")))["__func__"]
                code = dict(inspect.getmembers(func))["__code__"]
                callable_name = getattr(f, "__class__").__qualname__
            filename, lineno = code.co_filename, code.co_firstlineno

        return callable_name, filename, lineno

    def forward(self, *args):
        def load_arg(a, env):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        logger.info("start to interpret graph")

        args_iter = iter(args)
        aipiler_env: Dict[str, Any] = {}

        graph_output: Optional[Any] = None

        for idx, node in enumerate(self.graph.nodes):
            assert isinstance(node, torch.fx.Node)
            logger.debug(f"interpreting node {idx}: {node.format_node()}")

            if node.op == "placeholder":
                arg = next(args_iter)
                assert isinstance(
                    arg, FakeTensor
                ), "input tensor must be aipiler Tensor"
                aipiler_env[node.name] = arg
            elif node.op == "get_attr":
                target_atoms = node.target.split(".")
                attr = self.graph_module
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr, atom):
                        raise RuntimeError(
                            f"Node referenced nonexistent target {target_atoms[:i]}"
                        )
                    attr = getattr(attr, atom)
                aipiler_env[node.name] = (
                    from_torch_to_fake_tensor(attr)
                    if isinstance(attr, torch.Tensor)
                    else attr
                )
            elif node.op == "call_function":
                exec_func = self._lookup_aipiler_function(node.target)
                aipiler_args = load_arg(node.args, aipiler_env)
                aipiler_kwargs = load_arg(node.kwargs, aipiler_env)
                try:
                    aipiler_env[node.name] = exec_func(*aipiler_args, **aipiler_kwargs)
                    # TODO: handle setitem
                    # from .register_functions import setitem

                    # if exec_func.functions[0] is setitem:
                    #     aipiler_env[str(node.args[0])] = aipiler_env[node.name]
                except Exception as e:
                    raise e
            elif node.op == "output":
                graph_output = aipiler_env[node.name] = load_arg(
                    node.args[0], aipiler_env
                )
            else:
                assert False

        logger.info("finish interpreting graph")
        return graph_output
