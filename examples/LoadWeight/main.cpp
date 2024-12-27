
#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/embed.h>
#include <sstream>

namespace py = pybind11;

// 日志级别枚举
enum class LogLevel { INFO, WARNING, ERROR };

// 日志输出函数
void log(LogLevel level, const std::string &message) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

  const char *level_str;
  switch (level) {
  case LogLevel::INFO:
    level_str = "INFO";
    break;
  case LogLevel::WARNING:
    level_str = "WARNING";
    break;
  case LogLevel::ERROR:
    level_str = "ERROR";
    break;
  }

  std::cout << "[" << ss.str() << "][" << level_str << "] " << message
            << std::endl;
}

void load_model(const std::string &model_path) {

  log(LogLevel::INFO, "Initializing Python interpreter");
  py::scoped_interpreter guard{};

  log(LogLevel::INFO, "Importing Python module: load_model");
  py::module load_model_module = py::module::import("load_model");

  log(LogLevel::INFO, "Loading model weights from: " + model_path);
  py::dict model_weights =
      load_model_module.attr("load_model_weights")(model_path);
  log(LogLevel::INFO, "Successfully loaded model weights");

  // 打印权重统计信息
  size_t total_params = 0;
  log(LogLevel::INFO, "Model weights summary:");
  for (const auto &item : model_weights) {
    std::string key = item.first.cast<std::string>();
    py::list value = item.second.cast<py::list>();
    size_t param_count = value.size();
    total_params += param_count;

    std::stringstream ss;
    ss << "Layer: " << key << ", Parameters: " << param_count;
    log(LogLevel::INFO, ss.str());
  }

  log(LogLevel::INFO, "Total parameters: " + std::to_string(total_params));
}

int main() {
  // 输入模型文件路径
  std::string model_path = "/home/gaoshihao/project/Aipiler/examples/"
                           "LoadWeight/pytorch_model_00001-of-00004.bin";

  // 加载模型权重
  load_model(model_path);

  return 0;
}