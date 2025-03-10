.PHONY: clean Self_attention
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

Self_attention: 
	python SelfAttention.py > log
	PYTHONPATH=../../python ../../build/bin/genSelfAttention

run: Self_attention
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"../../thirdparty/llvm/build/lib" ./Self_attention

clean:
	rm -rf *.mlir *.ll Self_attention *.bin