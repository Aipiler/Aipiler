#include "Utils/loadPytorchModel.h"
#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "mix/mixTypes.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
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
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include "Utils/loadPytorchModel.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <sys/types.h>

// namespace py = pybind11;

// 日志级别枚举
// enum class LogLevel { INFO, WARNING, ERROR };

// 日志输出函数
// void log(LogLevel level, const std::string &message) {
//   auto now = std::chrono::system_clock::now();
//   auto time = std::chrono::system_clock::to_time_t(now);
//   std::stringstream ss;
//   ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

//   const char *level_str;
//   switch (level) {
//   case LogLevel::INFO:
//     level_str = "INFO";
//     break;
//   case LogLevel::WARNING:
//     level_str = "WARNING";
//     break;
//   case LogLevel::ERROR:
//     level_str = "ERROR";
//     break;
//   }

//   std::cout << "[" << ss.str() << "][" << level_str << "] " << message
//             << std::endl;
// }

// void load_model(const std::string model_path, mlir::ModuleOp &theModule,
//                 mlir::OpBuilder &builder) {
//   try {
//     log(LogLevel::INFO, "Initializing Python interpreter");
//     py::scoped_interpreter guard{};

//     log(LogLevel::INFO, "Importing Python module: load_model");
//     py::module load_model_module = py::module::import("load_model");
//     log(LogLevel::INFO, "Loading model weights from: " + model_path);
//     py::dict model_weights =
//         load_model_module.attr("load_model_weights")(model_path);
//     log(LogLevel::INFO, "Successfully loaded model weights");

//     for (auto item : model_weights) {
//       std::string key = item.first.cast<std::string>();
//       py::array value = item.second.cast<py::array>();
//       auto value_shape = value.shape();
//       auto rank = value.ndim();
//       llvm::SmallVector<int64_t> shape;
//       for (int i = 0; i < rank; i++) {
//         shape.push_back(value_shape[i]);
//       }
//       auto value_dtype = value.dtype();
//       // TODO: change dtype according to  value_dtype
//       auto dtype = builder.getF32Type();
//       auto tensorTy = mlir::RankedTensorType::get(shape, dtype);
//       const void *data = value.data();
//       const char *raw_data = static_cast<const char *>(data);
//       auto size = value.size();
//       auto raw_data_byte_len =
//           size * (dtype.getIntOrFloatBitWidth() / 8) / sizeof(char);

//       auto tensorAttr = mlir::DenseElementsAttr::getFromRawBuffer(
//           tensorTy, llvm::ArrayRef<char>(raw_data, raw_data_byte_len));
//       theModule->setAttr(key, tensorAttr);
//     }
//   } catch (const py::error_already_set &e) {
//     log(LogLevel::ERROR, "Python exception: " + std::string(e.what()));
//     throw;
//   } catch (const std::exception &e) {
//     log(LogLevel::ERROR, "Standard exception: " + std::string(e.what()));
//     throw;
//   } catch (...) {
//     log(LogLevel::ERROR, "Unknown exception occurred");
//     throw;
//   }
// }

using namespace mlir;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass();
std::unique_ptr<Pass> createLowerPrimaryToTosa();

void registerLowerModulePass();
void registerLowerCompositePass();
void registerLowerPrimaryToTosaPass();

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder) {
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(theModule.getBody());
  // printMemrefF32
  auto elementType = builder.getF32Type();
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printfunc =
      builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  printfunc.setPrivate();

  // Graph0
  auto inputType = RankedTensorType::get({2, 10}, builder.getF32Type());
  //   auto resultType
  auto functionTy = builder.getFunctionType({inputType}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto input = graph0.getArgument(0);
  auto linear0 = builder.create<mix::LinearOp>(loc, input, "linear", 10, 5,
                                               true, builder.getF32Type());

  auto returnType = linear0.getType();
  graph0.setFunctionType(builder.getFunctionType(inputType, returnType));
  builder.create<func::ReturnOp>(loc, ValueRange{linear0});

  // Main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  SmallVector<float> inputdata;
  for (int i = 0; i < 20; i++) {
    inputdata.push_back(float(i));
  }

  auto argAttr =
      DenseElementsAttr::get(inputType, llvm::ArrayRef<float>(inputdata));

  auto arg0 = builder.create<arith::ConstantOp>(loc, argAttr);
  auto call0 = builder.create<func::CallOp>(loc, graph0, ValueRange{arg0});
  auto cast =
      builder.create<tensor::CastOp>(loc, printInputType, call0->getResult(0));
  builder.create<func::CallOp>(loc, printfunc, ValueRange{cast});
  builder.create<func::ReturnOp>(loc);
}

int main() {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  registerLowerModulePass();
  registerLowerCompositePass();
  registerLowerPrimaryToTosaPass();

  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  generateCode(theModule, builder);

  load_model("./linear_model.bin", theModule, builder, builder.getF32Type());

  // mlir::PassManager pm(&context);
  // pm.addPass(createLowerModulePass());
  // pm.addPass(createLowerCompositePass());
  // pm.addPass(createLowerPrimaryToTosa());

  // // Add the lower pass
  // llvm::StringRef passPipelineStr = "builtin.module( \
  //       lower-mix-module, \
  //       lower-mix-composite, \
	// 	lower-mix-primary-to-tosa, \
  //       func.func( \
  //           tosa-to-linalg-named, \
  //           tosa-to-linalg, \
  //           tosa-to-tensor \
  //       ), \
  //       empty-tensor-to-alloc-tensor, \
  //       one-shot-bufferize{bufferize-function-boundaries}, \
  //       convert-linalg-to-loops, \
  //       buffer-deallocation-pipeline, \
  //       expand-strided-metadata, \
  //       lower-affine, \
  //       lower-affine, \
  //       finalize-memref-to-llvm, \
  //       convert-math-to-llvm, \
  //       convert-scf-to-cf, \
  //       convert-cf-to-llvm, \
  //       convert-func-to-llvm, \
  //       reconcile-unrealized-casts \
  //   )";

  // if (mlir::failed(mlir::parsePassPipeline(passPipelineStr, pm))) {
  //   std::cerr << "error happened when parse pass." << std::endl;
  //   return 1;
  // }
  // if (mlir::failed(pm.run(theModule))) {
  //   return 4;
  // }
  theModule->dump();

  return 0;
}