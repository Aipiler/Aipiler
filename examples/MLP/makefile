.PHONY: clean mlp
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

mlp: 
	python MLP.py
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genMLP

run_mlp: mlp
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./mlp

clean:
	rm -rf *.mlir *.ll mlp *.bin