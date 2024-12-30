MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../thirdparty/llvm/build/bin/mlir-opt


# TODO: add lower pass for self-attention op
lower-rotary-embedding:
	@${MIX_OPT} ./RotaryEmbedding.mlir \
		-lower-mix-primary-to-tosa \
		-o ./log.mlir

.PHONY: clean

clean:
	rm log.mlir