## Building steps
### Clone and Initialize

```
$ git clone git@github.com:Aipiler/Aipiler.git
$ cd Aipiler
$ git submodule update --init
```

### Create your conda env
```
$ conda create -n aipiler python=3.12
$ pip install -r requirements.txt
```

### Build and Test LLVM/MLIR/CLANG with python binding


```shell
$ cd thirdparty/llvm
$ git apply ../llvm_lld_link.patch
$ cmake -G Ninja -Sllvm -Bbuild \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ cd build && ninja check-mlir check-clang check-lld

$ cd thirdpart/llvm
$ export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core:${PYTHONPATH} 
```


### Build Aipiler

```shell
$ cmake -G Ninja -S. -Bbuild \
    -DMLIR_DIR=$PWD/thirdparty/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/thirdparty/llvm/build/lib/cmake/llvm \
    -DLLD_DIR=$PWD/thirdparty/llvm/build/lib/cmake/lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ cmake --build build
```

