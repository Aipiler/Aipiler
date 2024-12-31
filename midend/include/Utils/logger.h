#ifndef MIX_LOGGER_H
#define MIX_LOGGER_H

#include <string>

namespace mix {
namespace utils {
// 日志级别枚举
enum class LogLevel { INFO, WARNING, ERROR };

// 日志输出函数
void log(LogLevel level, const std::string &message);

} // namespace utils
} // namespace mix

#endif