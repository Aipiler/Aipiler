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

std::unique_ptr<Pass> createLowerModulePass();

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());

  auto elementType = builder.getF32Type();
  auto tensorType = RankedTensorType::get({2, 2}, elementType);

  auto functionTy =
      builder.getFunctionType({tensorType, tensorType}, {tensorType});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states = graph0.getArgument(0);
  auto residual = graph0.getArgument(1);

  auto linear0 = builder.create<mix::LinearOp>(loc, hidden_states,
                                               "model_parameters.mlp.linear0",
                                               2, 2, false, elementType);
  auto silu0 = builder.create<mix::SiLUOp>(loc, linear0);

  auto linear1 = builder.create<mix::LinearOp>(loc, hidden_states,
                                               "model_parameters.mlp.linear1",
                                               2, 2, false, elementType);
  auto mul0 = builder.create<mix::MulOp>(loc, silu0, linear1);

  auto linear2 = builder.create<mix::LinearOp>(
      loc, mul0, "model_parameters.mlp.linear2", 2, 2, true, elementType);
  auto output = builder.create<mix::AddOp>(loc, linear2, residual);
  builder.create<func::ReturnOp>(loc, ValueRange{output});

  theModule->dump();

  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  if (mlir::failed(pm.run(theModule))) {
    return 4;
  }
  std::cout << "==== After Lower pass GRAPH-LOWER =====" << std::endl;
  theModule->dump();

  return 0;
}