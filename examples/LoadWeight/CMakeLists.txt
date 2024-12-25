

# 查找 Python 库和头文件
find_package(Python3 COMPONENTS Interpreter Development)

# 添加 C++ 源文件
add_executable(PyTorchCppExample main.cpp)

# 链接 pybind11 和 Python 库
target_link_libraries(PyTorchCppExample PRIVATE pybind11::embed Python3::Python)

# 设置 Python 模块路径
set(PYTHON_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
