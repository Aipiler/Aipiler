iree-benchmark=../../thirdparty/iree/build/tools/iree-benchmark-module

benchmark: mlp.bf
	${iree-benchmark} \
		--module=$^ \
		--device=local-task \
		--function=main \
		--input="2x2xf32=[1, 2, 3, 4]" \
		--input="2x2xf32=[1, 2, 3, 4]"
	
mlp.bf: mlp_benchmark.py
	python $^

clean:
	rm -rf *.vmfb