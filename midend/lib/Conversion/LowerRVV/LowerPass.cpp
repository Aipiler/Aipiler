#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "rvv/rvvDialect.h"
#include "rvv/Transforms.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class LowerRVVToLLVMPass
    : public PassWrapper<LowerRVVToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRVVToLLVMPass)
  StringRef getArgument() const final { return "lower-rvv"; }
  StringRef getDescription() const final {
    return "RVV dialect lowering pass.";
  }
  LowerRVVToLLVMPass() = default;
  LowerRVVToLLVMPass(const LowerRVVToLLVMPass &) {}

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<rvv::RVVDialect>();
  }

  Option<bool> isOnRV32{*this, "rv32",
                        llvm::cl::desc("Emit RVV intrinsics on rv32"),
                        llvm::cl::init(false)};

  void runOnOperation() override;
};
} // namespace

void LowerRVVToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  LLVMConversionTarget target(*context);

  int64_t RVVIndexBitwidth;
  if (isOnRV32)
    RVVIndexBitwidth = 32;
  else
    RVVIndexBitwidth = 64;
  configureRVVLegalizeForExportTarget(target);
  populateRVVLegalizeForLLVMExportPatterns(converter, patterns, RVVIndexBitwidth);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void registerLowerRVVPass() { 
  PassRegistration<LowerRVVToLLVMPass>(); 
}

