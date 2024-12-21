MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
CLANGXX := ../../llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../llvm/build/lib
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

gelu_prim.mlir: gelu.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite \
		)" -o $@

.PHONY: clean

clean:
	rm gelu_*