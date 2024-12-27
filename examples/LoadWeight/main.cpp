#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
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

void load_model(const std::string &model_path, mlir::ModuleOp &theModule,
                mlir::OpBuilder &builder) {
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
      // TODO: change dtype according to  value_dtype
      auto dtype = builder.getF16Type();
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

int main() {
  // 输入模型文件路径
  std::string model_path = "pytorch_model_00001-of-00004.bin";

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());
  // theModule->setAttr("abc", builder.getI32TensorAttr({1, 2, 3, 4}));
  // auto abc = theModule->getAttr("abcd");
  // if (abc) {
  //   abc.dump();

  // } else {
  //   std::cout << "abccccc" << std::endl;
  // }
  // // 加载模型权重
  load_model(model_path, theModule, builder);
  theModule->dump();
  return 0;
}