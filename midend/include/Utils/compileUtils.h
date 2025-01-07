#ifndef MIDEND_UTILS_CPMPILEUTILS_H
#define MIDEND_UTILS_CPMPILEUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
namespace mix {
namespace utils {
class CompileUtils {
public:
  enum class TARGET { MLIR, LLVMIR, OBJECT, EXECUTE };
  CompileUtils(mlir::ModuleOp theModule, std::string targetName, TARGET target);
  CompileUtils(mlir::ModuleOp theModule, std::string targetName,
               std::set<TARGET> targets);
  void compile();

private:
  // mlir module
  mlir::ModuleOp theModule;
  // output target name
  std::string targetName;
  // buffer of object file.
  llvm::SmallVector<char, 0> buffer;
  llvm::LLVMContext LLVMContext;
  std::unique_ptr<llvm::Module> llvmModule;
  std::set<TARGET> targets;
  void dumpMLIR(std::string);
  void MLIRToLLVMIR();
  void dumpLLVMIR(std::string);
  void LLVMIRToObjectBuffer();
  void dumpObject(std::string);
  void ObjectBufferToExecutable(std::string);
};

} // namespace utils
} // namespace mix

#endif