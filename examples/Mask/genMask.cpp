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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
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
#include <tuple>

using namespace mlir;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass();
std::unique_ptr<Pass> createLowerPrimaryToTosa();
const int max_seq_len = 40;

ArrayAttr createIntArrayAttr(MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

auto genMask(MLIRContext &context, OpBuilder &builder, Location loc,
             Value attention_mask) {
  auto attention_mask_type =
      mlir::dyn_cast<mlir::RankedTensorType>(attention_mask.getType());
  auto attention_mask_shape = attention_mask_type.getShape();
  auto seq_len = attention_mask_shape[1];

  // types:
  auto type_i1 = builder.getI1Type();
  auto type_i16 = builder.getI16Type();
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_f32 = builder.getF32Type();
  auto type_f16 = builder.getF16Type();
  auto type_f64 = builder.getF64Type();

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  // 逻辑

  // line 12 : torch.aten.empty.memory_format
  auto dense12 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len, seq_len}, type_i1), {false});
  auto constant12 = builder.create<mix::ConstantOp>(loc, dense12);

  // line 19 : torch.aten.arange
  SmallVector<int16_t> tmp19(seq_len);
  for (int i = 0; i < seq_len; i++) {
    tmp19[i] = i;
  }
  auto dense19 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len}, type_i16), ArrayRef<int16_t>(tmp19));
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense19);

  // line 25: torch.aten.slice.Tensor
  auto slice25 =
      builder.create<mix::SliceOp>(loc, constant94, 0, 0, INT64_MAX, 1);

  // line 28: torch.aten.unsqueeze
  auto unsqueeze28 = builder.create<mix::UnsqueezeOp>(loc, slice25, attr_i32_1);

  // line 31: torch.aten.unsqueeze
  auto unsqueeze31 =
      builder.create<mix::UnsqueezeOp>(loc, constant94, attr_i32_0);

  // line 37: torch.aten.slice.Tensor
  auto slice37 =
      builder.create<mix::SliceOp>(loc, unsqueeze31, 1, 0, INT64_MAX, 1);

  // line 39: torch.aten.lt.Tensor
  auto lt39 = builder.create<mix::LtOp>(loc, unsqueeze28, slice37);

  // line 45: torch.aten.slice.Tensor
  auto slice45 =
      builder.create<mix::SliceOp>(loc, constant12, 0, 0, INT64_MAX, 1);

  // line 51: torch.aten.slice.Tensor
  auto slice51 = builder.create<mix::SliceOp>(loc, slice45, 1, 0, INT64_MAX, 1);

  // line 54: torch.aten.copy 略过，直接使用lt39 代替%1

  // line 60: torch.aten.slice.Tensor
  auto slice60 = builder.create<mix::SliceOp>(loc, lt39, 1, 0, INT64_MAX, 1);

  // line 66: torch.aten.slice_scatter 略过

  // line 81: torch.aten.unsqueeze
  auto unsqueeze81 =
      builder.create<mix::UnsqueezeOp>(loc, attention_mask, attr_i32_1);

  // line 84: torch.aten.unsqueeze
  auto unsqueeze84 =
      builder.create<mix::UnsqueezeOp>(loc, unsqueeze81, attr_i32_2);

  // line 100: torch.aten.bitwise_not
  auto bitwise_not100 = builder.create<mix::BitwiseNotOp>(loc, unsqueeze84);

  // line 108: torch.aten.expand
  auto expand108 = builder.create<mix::ExpandOp>(
      loc, bitwise_not100, createIntArrayAttr(context, {1, 1, 5, 5}));

  // line 111: torch.aten.unsqueeze
  auto unsqueeze111 = builder.create<mix::UnsqueezeOp>(loc, lt39, attr_i32_0);

  // line 114: torch.aten.unsqueeze
  auto unsqueeze114 =
      builder.create<mix::UnsqueezeOp>(loc, unsqueeze111, attr_i32_1);

  // line 136: torch.aten.bitwise_or.Tensor
  auto bitwise_or136 =
      builder.create<mix::BitwiseOrOp>(loc, expand108, unsqueeze114);
  return bitwise_or136;
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

  auto elementType = builder.getI1Type();
  SmallVector<int64_t> shape{1, 5}; // seq_len = 5
  auto maskType = RankedTensorType::get(shape, elementType);

  SmallVector<int64_t> shape2{1, 1, 5, 5}; // seq_len = 5
  auto outputType = RankedTensorType::get(shape2, elementType);
  auto functionTy = builder.getFunctionType({maskType}, {outputType});
  auto graph0 = builder.create<func::FuncOp>(loc, "Mask", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto output =
      genMask(context, builder, graph0->getLoc(), graph0.getArgument(0));

  builder.create<func::ReturnOp>(loc, ValueRange{output});

  theModule->dump();

  std::cout << "-------------------------------------------------" << std::endl;

  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  pm.addPass(createLowerCompositePass());
  pm.addPass(createLowerPrimaryToTosa());

  if (mlir::failed(pm.run(theModule))) {
    return -1;
  }
  theModule->dump();

  return 0;
}