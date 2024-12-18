MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
CLANGXX := ../../llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../llvm/build/lib
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

rms_prim.mlir: rms.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite \
		)" -o $@
		
rms_tosa.mlir: rms_prim.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-primary-to-tosa \
		)" -o $@

rms.ll: rms_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			func.func( \
			tosa-to-linalg-named, \
			tosa-to-linalg, \
			tosa-to-tensor \
			), \
			empty-tensor-to-alloc-tensor, \
			arith-bufferize, \
			func.func(tensor-bufferize, linalg-bufferize ), \
			func-bufferize, \
			func.func(buffer-deallocation, convert-linalg-to-loops), \
			convert-scf-to-cf, \
			convert-math-to-llvm, \
			convert-func-to-llvm, \
			expand-strided-metadata, \
			finalize-memref-to-llvm, \
			reconcile-unrealized-casts \
		)" | \
	${MLIR_TRANSLATE} --mlir-to-llvmir -o $@

rms: rms.ll launch.cpp
	${CLANGXX} -L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils $^ -o $@

run_rms: rms
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./rms

.PHONY: clean

clean:
	rm rms_* *.ll rms