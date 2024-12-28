## Building steps
### Clone and Initialize

```
$ git clone git@github.com:Aipiler/Aipiler.git
$ cd Aipiler
$ git submodule update --init
```

### Build and Test LLVM/MLIR/CLANG

```
$ cd thirdparty/llvm
$ cmake -G Ninja -Sllvm -Bbuild \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ cd build && ninja check-mlir check-clang
```

### Build mix-mlir

```
$ cd mix-mlir
$ cmake -G Ninja -S. -Bbuild \
    -DMLIR_DIR=$PWD/thirdparty/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/thirdparty/llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ cmake --build build
```

### Create your conda env
```
$ conda create -n aipiler python=3.12
$ pip install -r requirements.txt
```