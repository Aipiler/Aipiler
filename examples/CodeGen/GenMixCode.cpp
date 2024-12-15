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

std::unique_ptr<Pass> createLowerGraphPass();

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());

  auto tensorType = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, builder.getF32Type());

  auto functionTy =
      builder.getFunctionType({tensorType, tensorType}, {tensorType});
  auto graph0 = builder.create<ml_program::SubgraphOp>(
      loc, "graph0", functionTy, ArrayAttr{}, ArrayAttr{},
      builder.getStringAttr("private"));
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states = graph0.getArgument(0);
  auto residual = graph0.getArgument(1);

  auto mlp_output =
      builder.create<mix::MLPOp>(loc, tensorType, hidden_states, residual);

  auto c1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  auto output = builder.create<mix::AddOp>(loc, tensorType, c1, mlp_output);

  builder.create<ml_program::OutputOp>(loc, ValueRange{output});

  theModule->dump();

  mlir::PassManager pm(&context);
  pm.addPass(createLowerGraphPass());
  if (mlir::failed(pm.run(theModule))) {
    return 4;
  }
  std::cout << "==== After Lower pass GRAPH-LOWER =====" << std::endl;
  theModule->dump();

  return 0;
}