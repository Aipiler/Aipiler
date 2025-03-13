.PHONY: clean Mask
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

print.o: print.cpp
	g++ $^ -c -o $@

Mask: ../../build/bin/genMask print.o
	python Mask.py > py.log
	PYTHONPATH=../../python ../../build/bin/genMask

run: Mask
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"../../thirdparty/llvm/build/lib" ./Mask > mlir.log

clean:
	rm -rf *.mlir *.ll Mask *.bin