# NOTICE: build target: mix-opt and genMLP first.

.PHONY: clean

MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
CLANGXX := ../../llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../llvm/build/lib
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

mlp.mlir:
	../../build/bin/genMLP 2> $@

mlp_tosa.mlir: mlp.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite, \
			lower-mix-primary-to-tosa, \
			view-op-graph \
		)" -o $@

mlp_tmp.mlir: mlp_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			one-shot-bufferize{bufferize-function-boundaries} \
		)" -o $@

mlp_memref.mlir: mlp_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			func.func( \
			tosa-to-linalg-named, \
			tosa-to-linalg, \
			tosa-to-tensor \
			), \
			one-shot-bufferize{bufferize-function-boundaries} \
			)" -o $@
# 			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize{bufferize-function-boundaries} \			
#			buffer-deallocation-pipeline, \
#			convert-linalg-to-loops \



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
			one-shot-bufferize{bufferize-function-boundaries}, \
			one-shot-bufferize{bufferize-function-boundaries}, \
 			convert-linalg-to-loops, \
			buffer-deallocation-pipeline, \
			expand-strided-metadata, \
			lower-affine, \
			lower-affine, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-scf-to-cf, \
			convert-cf-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o $@

mlp: mlp.ll
	${CLANGXX} -L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils $^ -o $@

run_mlp: mlp
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./mlp



clean:
	rm -rf *.mlir *.ll mlp