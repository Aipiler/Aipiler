MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt
MLIR_CPU_RUNNER := ../../llvm/build/bin/mlir-cpu-runner
MLIR_TRANSLATE := ../../llvm/build/bin/mlir-translate
CLANGXX := ../../llvm/build/bin/clang++

OPT_FLAG := -O0
MLIR_LIB := ../../llvm/build/lib
MLIR_RUNNER_UTILS := ../../llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so

gelu_prim.mlir:
	../../build/bin/genGelu 2> $@


clean:
	rm gelu_*

gelu-mixtotosa-lower:
	${MIX_OPT} gelu_prim.mlir \
	--pass-pipeline="builtin.module( \
		lower-mix-primary-to-tosa, \
		func.func( \
		tosa-to-linalg-named, \
		tosa-to-linalg, \
		tosa-to-tensor \
		), \
		empty-tensor-to-alloc-tensor, \
		one-shot-bufferize{bufferize-function-boundaries}, \
		convert-math-to-libm, \
		convert-linalg-to-loops, \
		buffer-deallocation-pipeline, \
		expand-strided-metadata, \
		lower-affine, \
		finalize-memref-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		convert-scf-to-cf, \
		convert-cf-to-llvm, \
		convert-arith-to-llvm, \
		convert-func-to-llvm, \
		reconcile-unrealized-casts \
	)" \
	-o log.mlir

gelu-mixtotosa-translate:
	${MIX_OPT} ./gelu_prim.mlir \
    --pass-pipeline="builtin.module( \
      lower-mix-primary-to-tosa, \
      func.func( \
			tosa-to-linalg, \
      tosa-to-linalg-named, \
      tosa-to-tensor \
      ), \
			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize{bufferize-function-boundaries}, \
			convert-math-to-libm, \
			convert-linalg-to-loops, \
			buffer-deallocation-pipeline, \
			expand-strided-metadata, \
			lower-affine, \
			finalize-memref-to-llvm, \
			convert-scf-to-cf, \
			convert-cf-to-llvm, \
			convert-arith-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
    )" | \
  ${MLIR_TRANSLATE} --mlir-to-llvmir -o log.ll

gelu-mixtotosa-run:
	${MIX_OPT} ./gelu_prim.mlir \
    --pass-pipeline="builtin.module( \
      lower-mix-primary-to-tosa, \
      func.func( \
			tosa-to-linalg, \
      tosa-to-linalg-named, \
      tosa-to-tensor \
      ), \
			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize{bufferize-function-boundaries}, \
			convert-math-to-libm, \
			convert-linalg-to-loops, \
			buffer-deallocation-pipeline, \
			expand-strided-metadata, \
			lower-affine, \
			finalize-memref-to-llvm, \
			convert-scf-to-cf, \
			convert-cf-to-llvm, \
			convert-arith-to-llvm, \
			convert-func-to-llvm, \
			reconcile-unrealized-casts \
    )" | \
		${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}
