import Aipiler.aot as aot


# A minimal program, with no functions or variables.
class BasicModule(aot.CompiledModule): ...


# Create an instance of the program and convert it to MLIR.
from iree.compiler.ir import Context

instance = BasicModule(context=Context())
module_str = str(aot.CompiledModule.get_mlir_module(instance))

print(module_str)
# module @basic {
# }
