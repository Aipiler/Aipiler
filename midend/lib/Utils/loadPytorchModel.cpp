#include "Utils/loadPytorchModel.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <sys/types.h>

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

void load_model(const std::string model_path, mlir::ModuleOp &theModule,
                mlir::OpBuilder &builder, mlir::Type dtype) {
  try {
    log(LogLevel::INFO, "Initializing Python interpreter");
    py::scoped_interpreter guard{};

    log(LogLevel::INFO, "Importing Python module: load_model");
    py::module load_model_module = py::module::import("load_model");
    log(LogLevel::INFO, "Loading model weights from: " + model_path);
    py::dict model_weights =
        load_model_module.attr("load_model_weights")(model_path);
    log(LogLevel::INFO, "Successfully loaded model weights");

    for (auto item : model_weights) {
      std::string key = item.first.cast<std::string>();
      py::array value = item.second.cast<py::array>();
      auto value_shape = value.shape();
      auto rank = value.ndim();
      llvm::SmallVector<int64_t> shape;
      for (int i = 0; i < rank; i++) {
        shape.push_back(value_shape[i]);
      }
      auto value_dtype = value.dtype();
      auto tensorTy = mlir::RankedTensorType::get(shape, dtype);
      const void *data = value.data();
      const char *raw_data = static_cast<const char *>(data);
      auto size = value.size();
      auto raw_data_byte_len =
          size * (dtype.getIntOrFloatBitWidth() / 8) / sizeof(char);

      auto tensorAttr = mlir::DenseElementsAttr::getFromRawBuffer(
          tensorTy, llvm::ArrayRef<char>(raw_data, raw_data_byte_len));
      theModule->setAttr(key, tensorAttr);
    }
  } catch (const py::error_already_set &e) {
    log(LogLevel::ERROR, "Python exception: " + std::string(e.what()));
    throw;
  } catch (const std::exception &e) {
    log(LogLevel::ERROR, "Standard exception: " + std::string(e.what()));
    throw;
  } catch (...) {
    log(LogLevel::ERROR, "Unknown exception occurred");
    throw;
  }
}

// int main() {
//   // 输入模型文件路径
//   std::string model_path = "pytorch_model_00001-of-00004.bin";

//   mlir::MLIRContext context;
//   context.getOrLoadDialect<mlir::arith::ArithDialect>();
//   context.getOrLoadDialect<mlir::func::FuncDialect>();
//   context.getOrLoadDialect<mlir::tensor::TensorDialect>();

//   mlir::OpBuilder builder(&context);
//   auto loc = builder.getUnknownLoc();
//   auto theModule = mlir::ModuleOp::create(loc);
//   builder.setInsertionPointToEnd(theModule.getBody());

//   // 加载模型权重
//   load_model(model_path, theModule, builder);
//   theModule->dump();
//   return 0;
// }