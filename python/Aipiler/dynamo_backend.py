import torch
from Aipiler.interpreter import Interpreter
from Aipiler.tensor import FakeTensor, from_torch_to_fake_tensor
from Aipiler.dim import Dim
from Aipiler.graph import EinsumGraph
from typing import List, Union, Optional
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


def trace_from(
    outputs: Optional[List[FakeTensor]],
    inputs: Optional[Union[FakeTensor, List[FakeTensor]]] = None,
) -> EinsumGraph:

    if isinstance(outputs, FakeTensor):
        if outputs._trace is None:
            raise ValueError("trace_from expects symbol tensor(s)")
        outputs = [outputs]
    else:
        outputs = list(outputs)
        assert all(isinstance(v, FakeTensor) for v in outputs)

    if inputs is not None:
        if isinstance(inputs, FakeTensor):
            inputs = [inputs]
        else:
            inputs = list(inputs)
    return EinsumGraph(outputs, inputs).update_nodes()


def get_einsum_graph(interpreter: Interpreter, example_inputs):
    inputs: List[FakeTensor] = []
    # prepare tensor input of einsum graph
    for torch_input in example_inputs:
        if isinstance(torch_input, torch.Tensor):
            inputs.append(from_torch_to_fake_tensor(torch_input))

    outputs = interpreter(*inputs)

    return trace_from(outputs, inputs=inputs)
