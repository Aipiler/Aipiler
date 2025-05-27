import torch
from Aipiler.interpreter import Interpreter
from Aipiler.tensor import Tensor, from_torch_tensor
from Aipiler.dim import Dim
from Aipiler.graph import EinsumGraph, trace_from
from typing import List, Union
import torch


def compile(ep: torch.export.ExportedProgram) -> EinsumGraph:
    assert isinstance(ep, torch.export.ExportedProgram)

    graph_module = ep.graph_module
    example_inputs = ep.example_inputs[0]
    kwargs = ep.example_inputs[1]

    # get the interpreter for the subgraph
    interpreter = Interpreter(graph_module)

    einsum_graph, inputs, outputs = get_einsum_graph(interpreter, example_inputs)
    del interpreter
    print(str(einsum_graph))
    # TODO: compile and return Callable
    return einsum_graph


def aipiler_backend(graph_module, example_inputs, **kwargs):
    assert isinstance(graph_module, torch.fx.GraphModule)

    # get the interpreter for the subgraph
    interpreter = Interpreter(graph_module)

    einsum_graph = get_einsum_graph(interpreter, example_inputs)
    del interpreter
    print(str(einsum_graph))
    # TODO: compile and return Callable
    return graph_module.forward


def get_einsum_graph(interpreter: Interpreter, example_inputs):
    inputs: List[Tensor] = []
    # prepare tensor input of einsum graph
    for torch_input in example_inputs:
        if isinstance(torch_input, torch.Tensor):
            inputs.append(from_torch_tensor(torch_input))

    outputs = interpreter(*inputs)

    return trace_from(outputs, inputs=inputs)
