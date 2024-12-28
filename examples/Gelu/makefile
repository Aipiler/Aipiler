.PHONY: clean gelu
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

gelu: 
	python gelu.py
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genGelu

run_gelu: gelu
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./gelu

clean:
	rm -rf *.mlir *.ll gelu *.bin