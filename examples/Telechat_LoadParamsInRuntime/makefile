.PHONY: clean linear linear.o
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_LIB := ../../build/lib/
AIPILER_PY_PATH := ../../python

telechat.o:
	../../build/bin/genTelechat_dynamic_load_params

launch.o: launch.cpp 
	g++ -g -c $^ -o $@

memref.o: memref.cpp
	g++ -g -c $^ -o $@

telechat: launch.o telechat.o memref.o
	g++ $^ -L ${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -L ${AIPILER_LIB} -lloadPytorchModelShared -o $@

run_tele: 
	PYTHONPATH=${AIPILER_PY_PATH}:${PYTHONPATH} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB}:${AIPILER_LIB} ./telechat

clean:
	rm -rf *.mlir *.ll telechat *.o
