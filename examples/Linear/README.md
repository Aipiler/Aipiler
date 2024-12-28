# How to run this example?

1. make target linear.

```bash
$ cd build && ninja genLinear
```

2. activate aipiler conda env.

```bash
$ conda activate aipiler
```

2. run linear.py and generate pytorch model file: `linear_model.bin`.

```bash
$ cd ../examples/Linear
$ python linear.py
```

3. run `build/bin/genLinear` and generate executable file: `linear`.

```bash
$ PYTHONPATH=../../python ../../build/bin/genLinear
```

4. run executable file

```bash
$ LD_LIBRARY_PATH=../../thirdparty/llvm/build/lib/:${LD_LIBRARY_PATH} ./linear
```