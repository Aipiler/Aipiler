#include "Utils/compileUtils.h"
#include "Utils/logger.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include <system_error>
#include <utility>
namespace mutils = mix::utils;

mutils::CompileUtils::CompileUtils(mlir::ModuleOp theModule,
                                   std::string targetName, TARGET target)
    : theModule(theModule), targetName(std::move(targetName)),
      targets({target}) {}

mutils::CompileUtils::CompileUtils(mlir::ModuleOp theModule,
                                   std::string targetName,
                                   std::set<TARGET> targets)
    : theModule(theModule), targetName(std::move(targetName)),
      targets(targets) {}

void mutils::CompileUtils::dumpMLIR(std::string targetName) {
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(targetName, EC, llvm::sys::fs::OF_None);
  if (EC) {
    log(LogLevel::ERROR, EC.message());
  }
  this->theModule->print(outputFile);
  outputFile.close();
}

void mutils::CompileUtils::MLIRToLLVMIR() {
  this->llvmModule =
      mlir::translateModuleToLLVMIR(this->theModule, this->LLVMContext);
}

void mutils::CompileUtils::dumpLLVMIR(std::string targetName) {
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(targetName, EC, llvm::sys::fs::OF_None);
  if (EC) {
    log(LogLevel::ERROR, EC.message());
  }
  this->llvmModule->print(outputFile, nullptr);
  outputFile.close();
}

void mutils::CompileUtils::LLVMIRToObjectBuffer() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  llvm::Module &module = *(this->llvmModule.get());
  module.setTargetTriple(targetTriple);
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    log(LogLevel::ERROR, error);
    return;
  }
  llvm::TargetOptions opt;
  llvm::TargetMachine *targetMachine =
      target->createTargetMachine(targetTriple, "generic", "", opt,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Large);
  module.setDataLayout(targetMachine->createDataLayout());
  llvm::raw_svector_ostream objStream(this->buffer);

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, objStream, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    log(LogLevel::ERROR, "TargetMachine can't emit a file of this type");
    return;
  }

  pass.run(module);
  return;
}

void mutils::CompileUtils::dumpObject(std::string targetName) {
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(targetName, EC, llvm::sys::fs::OF_None);
  if (EC) {
    log(LogLevel::ERROR, EC.message());
  }

  outputFile.write(this->buffer.data(), buffer.size());
  outputFile.close();
}

namespace lld {
namespace elf {
bool link(llvm::MemoryBuffer *buffer, llvm::ArrayRef<const char *> args,
          llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS,
          bool exitEarly = false, bool disableOutput = false);
} // namespace elf
} // namespace lld

void mutils::CompileUtils::ObjectBufferToExecutable(std::string targetName) {
  // TODO: get args autoly.
  llvm::SmallVector<const char *, 32> args = {
      "ld.lld",
      "-z",
      "relro",
      "--hash-style=gnu",
      "--eh-frame-hdr",
      "-m",
      "elf_x86_64",
      "-pie",
      "-dynamic-linker",
      "/lib64/ld-linux-x86-64.so.2",
      "-o",
      targetName.data(),
      "/lib/x86_64-linux-gnu/Scrt1.o",
      "/lib/x86_64-linux-gnu/crti.o",
      "/usr/lib/gcc/x86_64-linux-gnu/11/crtbeginS.o",
      "-L../../thirdparty/llvm/build/lib",
      "-L/usr/lib/gcc/x86_64-linux-gnu/11",
      "-L/usr/lib/gcc/x86_64-linux-gnu/11/../../../../lib64",
      "-L/lib/x86_64-linux-gnu",
      "-L/lib/../lib64",
      "-L/usr/lib/x86_64-linux-gnu",
      "-L/usr/lib/../lib64",
      "-L/lib",
      "-L/usr/lib",
      "from_memory", // load from memory
      "-lgcc",
      "--as-needed",
      "-lgcc_s",
      "--no-as-needed",
      "-lc",
      "-lgcc",
      "-lm",
      "-lmlir_runner_utils",
      "-lmlir_c_runner_utils",
      "--as-needed",
      "-lgcc_s",
      "--no-as-needed",
      "/usr/lib/gcc/x86_64-linux-gnu/11/crtendS.o",
      "/lib/x86_64-linux-gnu/crtn.o"};
  auto objBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(this->buffer.data(), this->buffer.size()),
      "inMemoryObjectBuffer", false);

  auto flag = lld::elf::link(objBuffer.get(), args, llvm::outs(), llvm::errs());
  if (!flag) {
    log(LogLevel::ERROR, "Linking failed.");
  }
}

void mutils::CompileUtils::compile() {
  std::set<TARGET> targets = this->targets;
  if (targets.find(TARGET::MLIR) != targets.end()) {
    auto name = targetName + ".mlir";
    log(LogLevel::INFO, "Dump MLIR to " + name);
    this->dumpMLIR(name);
    targets.erase(TARGET::MLIR);
    if (targets.size() == 0)
      return;
  }
  log(LogLevel::INFO, "Start to translate MLIR to LLVMIR.");
  this->MLIRToLLVMIR();
  log(LogLevel::INFO, "Finish translatting MLIR to LLVMIR.");
  if (targets.find(TARGET::LLVMIR) != targets.end()) {
    auto name = targetName + ".ll";
    log(LogLevel::INFO, "Dump LLVM IR to " + name);
    this->dumpLLVMIR(name);
    targets.erase(TARGET::LLVMIR);
    if (targets.size() == 0)
      return;
  }
  log(LogLevel::INFO, "Start to dump LLVMIR to Memory Buffer.");
  this->LLVMIRToObjectBuffer();
  log(LogLevel::INFO, "Finish dumping LLVMIR to Memory Buffer.");
  if (targets.find(TARGET::OBJECT) != targets.end()) {
    auto name = targetName + ".o";
    log(LogLevel::INFO, "Dump Object to " + name);
    this->dumpObject(name);
    targets.erase(TARGET::LLVMIR);
    if (targets.size() == 0)
      return;
  }
  if (targets.find(TARGET::EXECUTE) != targets.end()) {
    log(LogLevel::INFO,
        "Start to link object from Memory Buffer to generate executable.");
    this->ObjectBufferToExecutable(this->targetName);
    log(LogLevel::INFO,
        "Finish linking object from Memory Buffer to generate executable.");
  }
}