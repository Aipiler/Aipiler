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


    def __str__(self) -> str:
        tensors = []
        def nameof(t):
            if t not in tensors:
                assert False
            return "t" + str(tensors.index(t))
        
        doc = "Graph("
        tensors += self.inputs
        param_doc = []
        for inp in self.inputs:
            n = nameof(inp)
            param_doc.append(n)
        doc += ", ".join(param_doc)
        doc += ")\n"
        
        for prim in self.nodes:
            if isinstance(prim, Map):
                lhs = prim.inputs[0]
                rhs = prim.inputs[1]
                ret = prim.output
                tensors.append(ret)
                prim_doc = "{ret} = map({lhs}, {rhs}, \"{einsum_str}\", [{map_dims}], \"{op}\")".format(
                    ret=nameof(ret), lhs=nameof(lhs), rhs=nameof(rhs), einsum_str=prim.einsum_str, map_dims=", ".join(["\"{}\"".format(letter) for letter in prim.ranks_to_map]), op=prim.op.value[1]
                )
                doc += "\t"
                doc += prim_doc
            elif isinstance(prim, Reduce):
                inp = prim.inputs[0]
                ret = prim.output
                tensors.append(ret)
                prim_doc = "{ret} = reduce({inp}, \"{einsum_str}\", \"{reduce_dims}\", \"{op}\")".format(
                    ret=nameof(ret), inp=nameof(inp), einsum_str=prim.einsum_str, reduce_dims=prim.reduce_rank, op=prim.op.value[1]
                )
                doc += "\t"
                doc += prim_doc
            else:
                doc += prim.__str__
            doc += "\n"
        doc += "\treturn "
        outputs = [nameof(out) for out in self.outputs]
        doc += ", ".join(outputs)
        return doc



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

