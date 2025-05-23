from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, Sequence
from Aipiler.tensor import Tensor
from Aipiler.primitive import EinsumPrimitive, Map, Reduce

class EinsumGraph:
    """节点管理器，负责管理计算图中的所有节点"""

    def __init__(self, outputs: Sequence[Tensor], inputs: Optional[Sequence[Tensor]] = None):
        self.outputs = list(outputs)
        self.inputs: Optional[List[Tensor]] = list(inputs) if inputs else []
        self.nodes: List[EinsumPrimitive] = self.update_nodes()

    def update_nodes(self):
        nodes : List[EinsumPrimitive] = []
        stack: List[EinsumPrimitive] = [output._trace for output in self.outputs]
        while stack:
            op = stack.pop()
            nodes.insert(0, op)
            for i in op.inputs:
                if i._trace:
                    stack.append(i._trace)
                else:
                    if i not in self.inputs:
                        self.inputs.append(i)
        return nodes


def trace_from(
    tensors: Optional[List[Tensor]], inputs: Optional[Union[Tensor, List[Tensor]]] = None
) -> EinsumGraph:
    
    if isinstance(tensors, Tensor):
        if tensors._trace is None:
            raise ValueError('trace_from expects symbol tensor(s)')
        outputs = [tensors]
    else:
        outputs = list(tensors)
        assert all(isinstance(v, Tensor) for v in outputs)

    if inputs is not None:
        if isinstance(inputs, Tensor):
            inputs = [inputs]
        else:
            inputs = list(inputs)
    return EinsumGraph(tensors, inputs)

