MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt

mlp-lower:
	@${MIX_OPT} ./mlp.mlir --lower-graph -o ./log.mlir

.PHONY: clean

clean:
	rm log.mlir