import torch
from Aipiler.interpreter import Interpreter
from Aipiler.tensor import Tensor, from_torch
from Aipiler.dim import Dim
from Aipiler.graph import EinsumGraph
from typing import List, Union, Optional


class Runtime:
    """
    Runime is responsible for compiling and executing the model.
    It takes an ExportedProgram and compiles it into an EinsumGraph.
    The EinsumGraph can then be executed with input data.
    """

    def __init__(self, exported_program: torch.export.ExportedProgram):
        assert isinstance(exported_program, torch.export.ExportedProgram)
        self.debug_mode = False
        self.exported_program = exported_program
        self.interpreter: Interpreter = Interpreter(exported_program.graph_module)
        self.einsum_graph: Union[EinsumGraph, None] = None

    def _trace_from(
        self,
        outputs: Optional[List[Tensor]],
        inputs: Optional[Union[Tensor, List[Tensor]]] = None,
    ) -> EinsumGraph:

        if isinstance(outputs, Tensor):
            if outputs._trace is None:
                raise ValueError("trace_from expects symbol tensor(s)")
            outputs = [outputs]
        else:
            outputs = list(outputs)
            assert all(isinstance(v, Tensor) for v in outputs)

        if inputs is not None:
            if isinstance(inputs, Tensor):
                inputs = [inputs]
            else:
                inputs = list(inputs)
        return EinsumGraph(outputs, inputs)

    def _get_einsum_graph(self, example_inputs) -> EinsumGraph:
        inputs: List[Tensor] = []
        # prepare tensor input of einsum graph
        for torch_input in example_inputs:
            if isinstance(torch_input, torch.Tensor):
                inputs.append(from_torch(torch_input))

        outputs = self.interpreter(*inputs)

        return self._trace_from(outputs, inputs=inputs)

    # TODO: execute接收pytorch的输入数据，返回pytorch的输出数据
    def execute(self, input_data):
        """
        Execute the model with the provided input data.
        """
        pass

    # TODO: compile 函数应该返回一个python内的可调用对象
    def compile(
        self,
    ):
        example_inputs = self.exported_program.example_inputs[0]
        kwargs = self.exported_program.example_inputs[1]

        self.einsum_graph = self._get_einsum_graph(example_inputs)
        print(str(self.einsum_graph))
        # TODO: compile and return Callable
        return self.einsum_graph

    # TODO: codegen 函数应该返回一个字符串或输出文件，表示编译后的代码。
    def codegen(self, file_path: str) -> str:
        pass
