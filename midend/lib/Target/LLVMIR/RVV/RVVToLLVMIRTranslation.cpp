#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
// #include "backend/include/llvm/IR/IntrinsicsRISCV.h"

#include "rvv/rvvDialect.h"
#include "Target/LLVMIR/Dialect/RVV/RVVToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the RVV dialect to LLVM IR.
class RVVDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "rvv/rvvConversions.inc"

    return failure();
  }
};
} // end namespace

void registerRVVDialectTranslation(DialectRegistry &registry) {
  registry.insert<rvv::RVVDialect>();
  registry.addExtension(+[](MLIRContext *ctx, rvv::RVVDialect *dialect) {
    dialect->addInterfaces<RVVDialectLLVMIRTranslationInterface>();
  });
}

void registerRVVDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerRVVDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
