.PHONY: clean embedding
MLIR_LIB := ../../thirdparty/llvm/build/lib
AIPILER_PY_PATH := ../../python

embedding: 
	python embedding.py
	PYTHONPATH=${AIPILER_PY_PATH} ../../build/bin/genEmbedding

run_embedding: embedding
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLIR_LIB} ./embedding

clean:
	rm -rf *.mlir *.ll embedding *.bin