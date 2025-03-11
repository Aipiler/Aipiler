.PHONY: clean rms
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

rms: 
	python mixedFusedRMSNorm.py
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genMixedFusedRMSNorm

run_rms: rms
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./rms

clean:
	rm -rf *.mlir *.ll rms *.bin