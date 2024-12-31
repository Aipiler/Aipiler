#include "Utils/logger.h"

#include <chrono>
#include <iomanip>
#include <iostream>

void mix::utils::log(LogLevel level, const std::string &message) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

  const char *level_str;
  switch (level) {
  case LogLevel::INFO:
    level_str = "INFO";
    break;
  case LogLevel::WARNING:
    level_str = "WARNING";
    break;
  case LogLevel::ERROR:
    level_str = "ERROR";
    break;
  }

  std::cout << "[" << ss.str() << "][" << level_str << "] " << message
            << std::endl;
}