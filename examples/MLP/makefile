MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt

mlp-lower:
	${MIX_OPT} ./mlp.mlir --lower-mix-module -o ./log.mlir

mlp-ml-opt:
	@${MIX_OPT} ./mlp.mlir --lower-mix-module --mlprogram-pipeline-globals -o ./log0.mlir

.PHONY: clean

clean:
	rm log*.mlir