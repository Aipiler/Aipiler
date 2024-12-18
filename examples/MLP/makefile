MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
CLANGXX := ../../llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../llvm/build/lib
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

mlp_tosa.mlir: mlp.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite, \
			lower-mix-primary-to-tosa \
		)" -o $@

mlp_memref.mlir: mlp_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			func.func( \
			tosa-to-linalg-named, \
			tosa-to-linalg, \
			tosa-to-tensor \
			), \
			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize, \
			buffer-deallocation-pipeline, \
			convert-linalg-to-loops \
		)" -o $@

mlp.ll: mlp_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-primary-to-tosa, \
			func.func( \
			tosa-to-linalg-named, \
			tosa-to-linalg, \
			tosa-to-tensor \
			), \
			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize, \
 			convert-linalg-to-loops, \
			buffer-deallocation-pipeline, \
			expand-strided-metadata, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-scf-to-cf, \
			convert-cf-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o $@

mlp: mlp.ll launch.cpp
	${CLANGXX} -L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils $^ -o $@

run_mlp: mlp
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./mlp

.PHONY: clean

clean:
	rm mlp_* *.ll mlp