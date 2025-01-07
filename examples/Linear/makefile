.PHONY: clean linear linear.o
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

linear.o:
	../../build/bin/genLinear

launch.o: launch.cpp
	g++ -c $^ -o $@

linear: launch.o linear.o
	g++ $^ -L ${MLIR_LIB} -lmlir_runner_utils -lmlir_c_runner_utils  -o $@

# linear: 
# 	python linear.py
# 	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genLinear

run_linear: linear
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./linear

clean:
	rm -rf *.mlir *.ll linear *.bin *.o