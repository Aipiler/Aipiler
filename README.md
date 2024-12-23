## Building steps
### Clone and Initialize

```
$ git clone git@github.com:Aipiler/Aipiler.git
$ cd Aipiler
$ git submodule update --init
```

### Build and Test LLVM/MLIR/CLANG

```
$ cd mix-mlir/llvm
$ git apply ../patch/ml_program_globalop_bufferize.patch
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
```

### Build mix-mlir

```
$ cd mix-mlir
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
```