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
const int seq_len = 40;

ArrayAttr createIntArrayAttr(MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

std::pair<Value, Value> genRotaryEmbedding(mlir::MLIRContext &context,
                                           mlir::OpBuilder &builder,
                                           Location loc) {

  // hidden_states dims
  auto hidden_size = 5120;
  auto n_head = 32;
  auto head_dim = hidden_size / n_head;
  auto key_value_projection_size = hidden_size * 2;
  auto key_value_projection_head_dim = key_value_projection_size / n_head;

  /* 定义一些可重用的信息 */

  // types:
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_f32 = builder.getF32Type();
  auto type_f64 = builder.getF16Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_key_value_weight =
      RankedTensorType::get({key_value_projection_size, hidden_size}, type_f32);
  auto type_dense_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_dense_bias = RankedTensorType::get({hidden_size}, type_f32);

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  /* 定义算子 */

  // 下面是RotaryEmbedding 的代码，应该在genRotaryEmbedding中实现

  // line 94: torch.aten.arange.start_step
  SmallVector<int64_t> tmp94(80);
  for (int i = 0, j = 0; i < 80; i++, j += 2) {
    tmp94[i] = j;
  }
  auto dense94 = DenseElementsAttr::get(RankedTensorType::get({80}, type_i64),
                                        ArrayRef<int64_t>(tmp94));
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense94);

  // line 102: torch.aten._to_copy
  auto convert102 = builder.create<mix::ConvertOp>(loc, constant94, type_f32);

  // line 104: torch.constant.int
  auto scalar101 = IntegerAttr::get(type_i64, 160);
  auto constant101 = builder.create<mix::ConstantOp>(loc, scalar101);

  // line 106: torch.aten.div.Tensor
  auto dev106 = builder.create<mix::DivOp>(loc, convert102, constant101);

  // line 108: torch.constant.float
  auto scalar108 = FloatAttr::get(type_f64, 30420.108888514722);
  auto constant108 = builder.create<mix::ConstantOp>(loc, scalar108);

  // line 110: torch.aten.pow.Scalar
  auto pow110 = builder.create<mix::PowOp>(loc, constant108, dev106);

  // line 112: torch.aten.reciprocal
  auto reciprocal112 = builder.create<mix::ReciprocalOp>(loc, pow110);

  // line 114: torch.constant.float
  auto scalar114 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant114 = builder.create<mix::ConstantOp>(loc, scalar114);

  // line 116: torch.aten.mul.Scalar
  auto mul116 = builder.create<mix::MulOp>(loc, constant114, reciprocal112);

  // line 123: torch.aten.arange
  SmallVector<float> tmp123(seq_len);
  for (int i = 0; i < seq_len; i++) {
    tmp123[i] = i;
  }
  auto dense123 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len}, type_f32), ArrayRef<float>(tmp123));
  auto constant123 = builder.create<mix::ConstantOp>(loc, dense123);

  // line 126: torch.aten.unsqueeze
  auto unsqueeze126 =
      builder.create<mix::UnsqueezeOp>(loc, constant123, attr_i32_1);

  // line 131: torch.aten.permute
  auto permute131 = builder.create<mix::PermuteOp>(
      loc, unsqueeze126, createIntArrayAttr(context, {0, 1}));

  // line 134: torch.aten.unsqueeze
  auto unsqueeze134 = builder.create<mix::UnsqueezeOp>(loc, mul116, attr_i32_1);

  // line 139: torch.aten.permute
  auto permute139 = builder.create<mix::PermuteOp>(
      loc, unsqueeze134, createIntArrayAttr(context, {1, 0}));

  // line 141: torch.aten.mul.Tensor
  auto mul141 = builder.create<mix::MulOp>(loc, permute131, permute139);

  // line 145: torch.aten.cat
  SmallVector<Value> tmp145{mul141, mul141};
  auto cat145 = builder.create<mix::ConcatOp>(
      loc, tmp145, IntegerAttr::get(IntegerType::get(&context, 64), 1));

  // line 147: torch.aten.cos
  auto cos147 = builder.create<mix::CosOp>(loc, cat145);

  // line 153: torch.aten.slice.Tensor
  auto slice148 = builder.create<mix::SliceOp>(loc, cos147, 0, 0, INT32_MAX, 1);

  // line 156: torch.aten.unsqueeze
  auto unsqueeze156 =
      builder.create<mix::UnsqueezeOp>(loc, slice148, attr_i32_1);

  // line 162: torch.aten.slice.Tensor
  auto slice162 =
      builder.create<mix::SliceOp>(loc, unsqueeze156, 2, 0, INT32_MAX, 1);

  // line 164: torch.constant.float
  auto scalar164 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant164 = builder.create<mix::ConstantOp>(loc, scalar164);

  // line 166: torch.aten.mul.Scalar
  auto mul166 = builder.create<mix::MulOp>(loc, slice162, constant164);

  // line 168: torch.aten.sin
  auto sin168 = builder.create<mix::SinOp>(loc, cat145);

  // line 174: torch.aten.slice.Tensor
  auto slice174 = builder.create<mix::SliceOp>(loc, sin168, 0, 0, INT32_MAX, 1);

  // line 177: torch.aten.unsqueeze
  auto unsqueeze177 =
      builder.create<mix::UnsqueezeOp>(loc, slice174, attr_i32_1);

  // line 183: torch.aten.slice.Tensor
  auto slice183 =
      builder.create<mix::SliceOp>(loc, unsqueeze177, 2, 0, INT32_MAX, 1);

  // line 185: torch.constant.float
  auto scalar185 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant185 = builder.create<mix::ConstantOp>(loc, scalar185);

  // line 187: torch.aten.mul.Scalar
  auto mul187 = builder.create<mix::MulOp>(loc, slice183, constant185);

  return {mul166, mul187};
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

  auto elementType = builder.getF32Type();
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printfunc =
      builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  printfunc.setPrivate();

  //
  SmallVector<int64_t> shape{8192, 1, 160};
  auto cosType = RankedTensorType::get(shape, elementType);
  auto sinType = RankedTensorType::get(shape, elementType);
  auto functionTy = builder.getFunctionType({}, {cosType, sinType});
  auto graph0 =
      builder.create<func::FuncOp>(loc, "RotaryEmbedding", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto [cos, sin] = genRotaryEmbedding(context, builder, graph0->getLoc());

  builder.create<func::ReturnOp>(loc, ValueRange{cos, sin});

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