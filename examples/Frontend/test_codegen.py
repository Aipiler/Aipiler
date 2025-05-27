# RUN: %PYTHON %s | mlir-opt -split-input-file | FileCheck
import mlir
import mlir.ir


# TODO: Move to a test utility class once any of this actually exists.
def print_module(f):
    m = f()
    print("// -----")
    print("// TEST_FUNCTION:", f.__name__)
    print(m.to_asm())
    return f


# CHECK-LABEL: TEST_FUNCTION: create_my_op
@print_module
def create_my_op():
    m = mlir.ir.Module()
    builder = m.new_op_builder()
    # CHECK: mydialect.my_operation ...
    builder.my_op()
    return m
