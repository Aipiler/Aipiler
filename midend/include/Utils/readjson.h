#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

bool getElementFromJson(std::string dataPath, std::vector<double> &result);