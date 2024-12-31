#include "Utils/loadPytorchModel.h"
#include "Utils/logger.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <sys/types.h>

namespace py = pybind11;
using namespace mix::utils;

void load_model(const std::vector<std::string> model_paths,
                mlir::ModuleOp &theModule, mlir::OpBuilder &builder,
                mlir::Type dtype) {
  try {
    log(LogLevel::INFO, "Initializing Python interpreter");
    py::scoped_interpreter guard{};
    log(LogLevel::INFO, "Importing Python module: Aipiler");
    py::module load_model_module = py::module::import("Aipiler");
    log(LogLevel::INFO, "Start Loadding model bin.");
    py::dict model_weights =
        load_model_module.attr("load_model_weights")(model_paths);
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
      log(LogLevel::INFO, "start create attr: " + key);
      auto tensorAttr = mlir::DenseElementsAttr::getFromRawBuffer(
          tensorTy, llvm::ArrayRef<char>(raw_data, raw_data_byte_len));
      theModule->setAttr(key, tensorAttr);
      log(LogLevel::INFO, "end create attr: " + key);
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

void load_model(const std::string model_path, mlir::ModuleOp &theModule,
                mlir::OpBuilder &builder, mlir::Type dtype) {
  try {
    log(LogLevel::INFO, "Initializing Python interpreter");
    py::scoped_interpreter guard{};
    log(LogLevel::INFO, "Importing Python module: Aipiler");
    py::module load_model_module = py::module::import("Aipiler");
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
      log(LogLevel::INFO, "start create attr: " + key);
      auto tensorAttr = mlir::DenseElementsAttr::getFromRawBuffer(
          tensorTy, llvm::ArrayRef<char>(raw_data, raw_data_byte_len));
      theModule->setAttr(key, tensorAttr);
      log(LogLevel::INFO, "end create attr: " + key);
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