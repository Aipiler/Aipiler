#!/bin/bash
MIX_OPT := ../../build/bin/mix-opt
MLIR_TRANSLATE := ../../thirdparty/llvm/build/bin/mlir-translate
MLIR_CPU_RUNNER := ../../thirdparty/llvm/build/bin/mlir-cpu-runner
LLC := ../../thirdparty/llvm/build/bin/llc
OPT_FLAG := -O0

MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ../../thirdparty/llvm/build/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

# cd ../../build && ninja
Telechat: print.o
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genTelechat

print.o: print.cpp
	g++ $^ -c -o $@

run_Telechat: Telechat print.o
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./Telechat

# LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../thirdparty/llvm/build/lib ./RotaryEmbedding

Telechat-run:
	@${MIX_OPT}  ./telechat.mlir \
		-pass-pipeline="builtin.module(lower-mix-module, lower-mix-composite, lower-mix-primary-to-tosa, func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith), \
		eliminate-empty-tensors, empty-tensor-to-alloc-tensor, one-shot-bufferize{bufferize-function-boundaries}, \
		convert-linalg-to-loops, convert-vector-to-scf, expand-strided-metadata, convert-vector-to-llvm, \
		arith-expand, lower-affine, convert-arith-to-llvm, convert-math-to-llvm, convert-math-to-libm, \
		finalize-memref-to-llvm, convert-scf-to-cf)" --mlir-print-ir-after-failure | \
	${MIX_OPT}	-llvm-request-c-wrappers | \
	${MIX_OPT}	-pass-pipeline="builtin.module(convert-arith-to-llvm, convert-math-to-llvm, convert-math-to-libm, convert-func-to-llvm, reconcile-unrealized-casts)" | \
	${MLIR_CPU_RUNNER} ${OPT_FLAG} -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}


.PHONY: clean 
clean:
	rm -rf *.mlir *.ll Telechat *.o