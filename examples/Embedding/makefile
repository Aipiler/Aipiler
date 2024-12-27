# NOTICE: build target: mix-opt and genMLP first.

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

embedding.mlir:
	../../build/bin/genEmbedding 2> $@

embedding_tosa.mlir: embedding.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			lower-mix-module, \
			lower-mix-composite, \
			lower-mix-primary-to-tosa, \
			view-op-graph \
		)" -o $@

embedding_memref.mlir: embedding_tosa.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
			func.func( \
			tosa-to-linalg-named, \
			tosa-to-linalg, \
			tosa-to-tensor \
			), \
			empty-tensor-to-alloc-tensor, \
			one-shot-bufferize{bufferize-function-boundaries} \
			)" -o $@



embedding.ll: embedding_memref.mlir
	${MIX_OPT} $^ \
		--pass-pipeline="builtin.module( \
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

embedding: embedding.ll
	${CLANGXX} -L${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils $^ -o $@

run_embed: embedding
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./embedding



clean:
	rm -rf *.mlir *.ll embedding