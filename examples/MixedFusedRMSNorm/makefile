# NOTICE: build target: mix-opt and genFusedRMSNorm first.

.PHONY: clean

MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../thirdparty/llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../thirdparty/llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../thirdparty/llvm/build/bin/mlir-translate
CLANGXX := ../../thirdparty/llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../thirdparty/llvm/build/lib
MLIR_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_c_runner_utils.so

rms.mlir:
	../../build/bin/genFusedRMSNorm 2> $@

rms_tosa.mlir: rms.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite, \
			lower-mix-primary-to-tosa \
		)" -o $@

rms.ll: rms_tosa.mlir
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
 			convert-linalg-to-loops, \
			buffer-deallocation-pipeline, \
			expand-strided-metadata, \
			lower-affine, \
			finalize-memref-to-llvm, \
			convert-math-to-llvm, \
			convert-scf-to-cf, \
			convert-cf-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o $@

rms: rms.ll
	${CLANGXX} -L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils $^ -o $@

run_rms: rms
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./rms



clean:
	rm -rf *.mlir *.ll rms