.PHONY: clean linear linear.o
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_LIB := ../../build/lib/
AIPILER_PY_PATH := ../../python

linear.o:
	../../build/bin/genLinear

launch.o: launch.cpp
	g++ -g -c $^ -o $@

linear: launch.o linear.o
	python linear.py
	g++ $^ -L ${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils -L ${AIPILER_LIB} -lloadPytorchModelShared -o $@

# linear: 
# 	python linear.py
# 	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genLinear

run_linear: linear
	PYTHONPATH=${AIPILER_PY_PATH}:${PYTHONPATH} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB}:${AIPILER_LIB} ./linear

clean:
	rm -rf *.mlir *.ll linear *.bin *.o