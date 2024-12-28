.PHONY: clean linear
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

linear: 
	python linear.py
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genLinear

run_linear: linear
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./linear

clean:
	rm -rf *.mlir *.ll linear *.bin