# How to run this example?

1. make target linear.

```bash
$ cd build && ninja genLinear
```

2. activate aipiler conda env.

```bash
$ conda activate aipiler
```

2. run linear.py and generate pytorch model file: `linear_model.bin`, run `build/bin/genLinear` and generate executable file: `linear`.


```bash
$ cd ../examples/Linear
$ make linear
```

3. run executable file

```bash
$ make run_linear
```

# How to load parameter dynamicly?

1. Cower WeighOp to load from global Memref pointer
Turn on option `DynamicLoadWeight` of pass `LowerCompositePass` in file `linear.cpp`:

```C++
pm.addPass(createLowerCompositePass(DynamicLoadWeight));
```
2. Create global Memref pointer and initize memref at runtime.

Create global memref pointer and initize in file `launch.cpp`:

```C++
// in global
RankedMemRefType<half, 2> *linear_weight;
RankedMemRefType<half, 1> *linear_bias;

...
// in main
  // initialize weight and bias
  half *linear_weight_data = new half[INPUT_SIZE * INPUT_SIZE];
  int64_t linear_weight_shape[2] = {INPUT_SIZE, INPUT_SIZE};

  half *linear_bias_data = new half[INPUT_SIZE];
  int64_t linear_bias_shape[1] = {INPUT_SIZE};

  linear_weight =
      new RankedMemRefType<half, 2>(linear_weight_data, linear_weight_shape);
  linear_bias =
      new RankedMemRefType<half, 1>(linear_bias_data, linear_bias_shape);
...
```
