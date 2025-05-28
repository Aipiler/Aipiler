from typing import Tuple
import functools
from mlir.dialects import func, memref, arith, builtin
from mlir.ir import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# func.func()
def gen_function() -> func.FuncOp:

    arguments = []
    shape1 = (3, ShapedType.get_dynamic_size())
    f32 = F32Type.get()
    shape2 = (ShapedType.get_dynamic_size(), 5)

    arguments.append(RankedTensorType.get(shape1, f32))
    arguments.append(RankedTensorType.get(shape2, f32))

    results = []
    shape3 = (3, 5)
    results.append(RankedTensorType.get(shape3, f32))

    function_type = FunctionType.get(inputs=arguments, results=results)

    op = func.FuncOp(name="main", type=function_type, visibility="private")
    return op


@run
def test():

    with Context() as ctx, Location.unknown():

        module = Module.create()

        with InsertionPoint(module.body):

            funcOp = gen_function()
        print(module)
