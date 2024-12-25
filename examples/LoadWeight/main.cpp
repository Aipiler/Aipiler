#include <iostream>
#include <pybind11/embed.h> // pybind11 的嵌入式 API
#include <string>
#include <vector>

namespace py = pybind11;

void load_model(const std::string &model_path) {
  try {
    // 初始化 Python 解释器
    py::scoped_interpreter guard{};

    // 导入 Python 脚本
    py::module load_model_module = py::module::import("load_model");

    // 调用 load_model_weights 函数
    py::dict model_weights =
        load_model_module.attr("load_model_weights")(model_path);

    // 输出模型权重的内容
    for (const auto &item : model_weights) {
      std::string key = item.first.cast<std::string>();
      py::list value = item.second.cast<py::list>();

      std::cout << "Key: " << key << std::endl;
      std::cout << "Values: ";
      for (const auto &val : value) {
        std::cout << val.cast<float>() << " ";
      }
      std::cout << std::endl;
    }
  } catch (const py::error_already_set &e) {
    std::cerr << "Python exception occurred: " << e.what() << std::endl;
  }
}

int main() {
  // 输入模型文件路径
  std::string model_path = "/home/gaoshihao/project/Aipiler/examples/"
                           "LoadWeight/pytorch_model_00001-of-00004.bin";

  // 加载模型权重
  load_model(model_path);

  return 0;
}
