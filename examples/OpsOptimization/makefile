Mix_BUILD_DIR := ../../build
LLVM_BUILD_DIR := ../../thirdparty/llvm/build/
Mix_OPT := ${Mix_BUILD_DIR}/bin/mix-opt
MLIR_TRANSLATE := ${LLVM_BUILD_DIR}/bin/mlir-translate
MLIR_CPU_RUNNER := ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner

ifeq ($(shell uname),Linux)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.so
MTRIPLE := x86_64-unknown-linux-gnu
else ifeq ($(shell uname),Darwin)
MLIR_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.dylib
MLIR_C_RUNNER_UTILS := ${LLVM_BUILD_DIR}/lib/libmlir_c_runner_utils.dylib
MTRIPLE := x86_64-apple-darwin
endif

matmul-vectorization-lower:
	@${Mix_OPT} ./matmul.mlir \
		-batchmatmul-optimize \
		-o ./log.mlir

matmul-vectorization-run:
	@${Mix_OPT} ./matmul.mlir \
		-batchmatmul-optimize \
		-convert-linalg-to-affine-loops \
		-lower-affine \
		-convert-vector-to-scf \
		-convert-scf-to-cf \
		-convert-vector-to-llvm \
		-convert-math-to-llvm \
		-convert-math-to-libm \
		-convert-arith-to-llvm \
		-convert-func-to-llvm \
		-expand-strided-metadata \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${MLIR_CPU_RUNNER} -O3 -e main -entry-point-result=void \
		-shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_C_RUNNER_UTILS}