## Building steps
### Clone and Initialize

```
$ git clone git@github.com:Aipiler/Aipiler.git
$ cd Aipiler
$ git submodule update --init
```

### Create your conda env
```shell
$ conda create -n aipiler python=3.12
$ cd thirdparty/iree
$ pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
$ cd ../../
$ pip install -r requirements.txt
```

### Build IREE with python binding 
```shell
$ cd thirdparty/iree
$ cmake -G Ninja -B build/ \
  -DIREE_TARGET_BACKEND_CUDA=ON \
  -DIREE_HAL_DRIVER_CUDA=ON \
  -DIREE_BUILD_PYTHON_BINDINGS=ON  \
  -DPython3_EXECUTABLE="$(which python3)" \
  .
$ cmake --build build
$ cmake --build build --target iree-run-tests
```

### Using the Python bindings

There are two available methods for installing the Python bindings, either through creating an editable wheel or through extending PYTHONPATH.

**Option A: Installing the bindings as editable wheels**

This method links the files in your build tree into your Python package directory as an editable wheel.
```shell
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e build/compiler
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e build/runtime
```

**Option B: Extending PYTHONPATH(Recommanded)**

This method more effectively captures the state of your build directory, but is prone to errors arising from forgetting to source the environment variables.

Extend your PYTHONPATH with IREE's bindings/python paths and try importing:

```shell
$ cp build/.env ../../.env
$ cd ../../
$ source .env && export PYTHONPATH
# The 'PYTHONPATH' environment variable should now contain
#   build/compiler/bindings/python;build/runtime/bindings/python
```

Test your Python bindings
```shell
python -c "import iree.compiler; help(iree.compiler)"
python -c "import iree.runtime; help(iree.runtime)"
```


<!-- 
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
``` -->

