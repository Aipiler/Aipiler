.PHONY: clean load
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

load: 
	python test_load.py
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/LoadWeight

run_load: load
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./rms

clean:
	rm -rf *.mlir *.ll load *.bin