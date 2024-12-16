MIX_OPT := ../../build/bin/mix-opt
MLIR_OPT := ../../llvm/build/bin/mlir-opt

mlp-lower:
	${MIX_OPT} ./mlp.mlir --lower-mix-module --lower-mix-composite --lower-mix-primary-to-tosa -o ./log.mlir

.PHONY: clean

clean:
	rm log*