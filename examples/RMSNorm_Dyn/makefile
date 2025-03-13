.PHONY: clean linear linear.o
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_LIB := ../../build/lib/
AIPILER_PY_PATH := ../../python

rms_norm.o:
	../../build/bin/genrms_dyn

launch.o: launch.cpp 
	g++ -g -c $^ -o $@

memref.o: memref.cpp
	g++ -g -c $^ -o $@

rms_norm: launch.o rms_norm.o memref.o
	g++ $^ -L ${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -L ${AIPILER_LIB} -lloadPytorchModelShared -o $@

run_rms: 
	python mixedFusedRMSNorm.py
	PYTHONPATH=${AIPILER_PY_PATH}:${PYTHONPATH} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB}:${AIPILER_LIB} ./rms_norm

clean:
	rm -rf *.mlir *.ll telechat *.o *.bin
