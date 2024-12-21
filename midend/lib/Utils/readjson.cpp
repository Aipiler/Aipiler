#include "Utils/readjson.h"

using namespace llvm;

// Helper function to split a string by a delimiter
std::vector<std::string> splitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// Recursive function to traverse JSON and find the array
const json::Array *findArray(const json::Object *obj,
                             const std::vector<std::string> &path,
                             size_t index = 0) {
  if (index >= path.size()) {
    return nullptr;
  }

  // Locate the next level in the JSON hierarchy
  auto iter = obj->find(path[index]);
  if (iter == obj->end()) {
    return nullptr;
  }

  // If it's the last key, return the array
  if (index == path.size() - 1) {
    return iter->second.getAsArray();
  }

  // Otherwise, continue searching the nested object
  if (const auto *nestedObj = iter->second.getAsObject()) {
    return findArray(nestedObj, path, index + 1);
  }

  return nullptr;
}

void flattenJSON(const json::Value &value, std::vector<double> &result) {
  if (const auto *array = value.getAsArray()) {
    for (const auto &elem : *array) {
      flattenJSON(elem, result);
    }
  } else if (value.getAsNumber().has_value()) {
    result.push_back(value.getAsNumber().value());
  } else {
    errs() << "Unsupported JSON element type encountered\n";
  }
}

bool getElementFromJson(std::string dataPath, std::vector<double> &result) {

  // Extract the path and split it
  std::string pathStr(dataPath);
  std::vector<std::string> path = splitString(pathStr, '.');
  std::string jsonPath = path[0] + ".json";
  path.erase(path.begin());

  // Read the JSON file
  std::ifstream jsonFile(jsonPath);
  if (!jsonFile.is_open()) {
    errs() << "Error: Cannot open file " << jsonPath << "\n";
    return false;
  }

  std::stringstream buffer;
  buffer << jsonFile.rdbuf();
  std::string jsonString = buffer.str();
  // Parse the JSON file
  Expected<json::Value> parsed = json::parse(jsonString);
  if (!parsed) {
    errs() << "Error: Failed to parse JSON: " << toString(parsed.takeError())
           << "\n";
    return false;
  }

  const auto *rootObj = parsed->getAsObject();
  if (!rootObj) {
    errs() << "Error: JSON root is not an object.\n";
    return false;
  }

  // Locate the array
  const auto *array = findArray(rootObj, path);
  if (!array) {
    errs() << "Error: Path does not lead to an array.\n";
    return false;
  }

  // write result
  for (auto v : *array) {
    flattenJSON(v, result);
  }
  return true;
}
