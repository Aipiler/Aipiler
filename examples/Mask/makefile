MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../thirdparty/llvm/build/bin/mlir-opt


# TODO: add lower pass for self-attention op
lower-self-attention:
	@${MIX_OPT} ./self_attention.mlir -o ./log.mlir

.PHONY: clean

clean:
	rm log.mlir