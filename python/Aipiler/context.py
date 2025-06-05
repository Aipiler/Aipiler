import torch
from Aipiler.interpreter import Interpreter
from Aipiler.tensor import FakeTensor, from_torch, from_torch_to_fake_tensor
from Aipiler.dim import Dim
from Aipiler.graph import EinsumGraph
from typing import List, Union, Optional


class Context:
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

    def compile(self) -> EinsumGraph:
        graph_module = self.exported_program.graph_module

        # get the interpreter for the subgraph
        interpreter = Interpreter(graph_module)

        einsum_graph = self.get_einsum_graph(interpreter)
        del interpreter
        # print(str(einsum_graph))
        # TODO: compile and return Callable
        return einsum_graph

    def trace_from(
        self,
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

    def get_einsum_graph(self, interpreter: Interpreter) -> EinsumGraph:
        inputs, outputs = interpreter()
        return self.trace_from(outputs, inputs=inputs)

    # TODO: execute接收pytorch的输入数据，返回pytorch的输出数据
    def execute(self, input_data):
        """
        Execute the model with the provided input data.
        """
        pass

    # TODO: codegen 函数应该返回一个字符串或输出文件，表示编译后的代码。
    def codegen(self, file_path: str) -> str:
        pass
