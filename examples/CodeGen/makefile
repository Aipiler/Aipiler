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

Telechat: 
	cd ../../build && ninja
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genTelechat

run_Telechat: Telechat
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./Telechat

# LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../thirdparty/llvm/build/lib ./RotaryEmbedding

.PHONY: clean 
clean:
	rm -rf *.mlir *.ll mlp *.bin Telechat