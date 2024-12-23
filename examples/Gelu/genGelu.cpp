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
  auto tensorType = RankedTensorType::get({2, 2}, elementType);

  auto functionTy = builder.getFunctionType({tensorType, tensorType}, {tensorType});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto input = graph0.getArgument(0);
  auto c5_n1 = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr(5.0e-1));
  auto mul0 = builder.create<mix::MulOp>(loc, input, c5_n1);
  auto cdot79788456 = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.79788456));
  auto mul1 = builder.create<mix::MulOp>(loc, input, cdot79788456);
  auto cdot044715 = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.044715));
  auto mul2 = builder.create<mix::MulOp>(loc, input, cdot044715);
  auto mul3 = builder.create<mix::MulOp>(loc, input, mul2);
  auto c1 =
      builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(1.0f));
  auto add0 = builder.create<mix::AddOp>(loc, c1, mul3);
  auto mul4 = builder.create<mix::MulOp>(loc, mul1, add0);
  auto tanh0 = builder.create<mix::TanhOp>(loc, mul4);
  auto add1 = builder.create<mix::AddOp>(loc, tanh0, c1);
  auto mul5 = builder.create<mix::MulOp>(loc, mul0, add1);
  builder.create<func::ReturnOp>(loc, ValueRange{mul5});

  // Main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  auto argAttr =
      DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>{1, 2, 3, 4});

  auto arg0 = builder.create<arith::ConstantOp>(loc, argAttr);
  auto arg1 = builder.create<arith::ConstantOp>(loc, argAttr);
  auto gelu = builder.create<func::CallOp>(loc, graph0, ValueRange{arg0, arg1});
  auto cast = builder.create<tensor::CastOp>(loc, printInputType, gelu.getResults()[0]);
  builder.create<func::CallOp>(loc, printfunc, ValueRange{cast});
  builder.create<func::ReturnOp>(loc);
  theModule->dump();

  return 0;
}