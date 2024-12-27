#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "mix/mixTypes.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace mlir;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass();
std::unique_ptr<Pass> createLowerPrimaryToTosa();

mlir::Value embedding(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value indices, std::string param_loc,
                      int num_embeddings, int embedding_dim, mlir::Type dtype) {
  auto embed0 =
      builder.create<mix::EmbeddingOp>(loc, indices, "model_parameters.embed",
                                       num_embeddings, embedding_dim, dtype);
  return embed0;
}

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
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
  llvm::SmallVector<int64_t> weightShape{4, 3};
  auto indicesType = RankedTensorType::get({2, 6}, builder.getI32Type());
  //   auto resultType
  auto functionTy = builder.getFunctionType({indicesType}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto indices = graph0.getArgument(0);
  auto embed0 = embedding(builder, loc, indices, "model_parameters.embed", 10,
                          4, builder.getF32Type());
  //   auto embed0 = builder.create<mix::EmbeddingOp>(
  //       loc, indices, "model_parameters.embed", 10, 4, builder.getF32Type());
  auto returnType = embed0.getType();
  graph0.setFunctionType(builder.getFunctionType(indicesType, returnType));
  builder.create<func::ReturnOp>(loc, ValueRange{embed0});

  // Main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  auto argAttr = DenseElementsAttr::get(
      indicesType, llvm::ArrayRef<int>{1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6});

  auto arg0 = builder.create<arith::ConstantOp>(loc, argAttr);
  auto call0 = builder.create<func::CallOp>(loc, graph0, ValueRange{arg0});
  auto cast =
      builder.create<tensor::CastOp>(loc, printInputType, call0->getResult(0));
  builder.create<func::CallOp>(loc, printfunc, ValueRange{cast});
  builder.create<func::ReturnOp>(loc);
  theModule->dump();

  return 0;
}