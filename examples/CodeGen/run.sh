#!/bin/bash

# 确保脚本在出错时停止
set -e

# 定义变量
BUILD_DIR="../../build"
GEN_TELECHAT_EXECUTABLE="./bin/genTelechat"
TELECHAT_EXECUTABLE="./Telechat"

# 切换到 build 目录
if [ -d "$BUILD_DIR" ]; then
    echo "切换到构建目录：$BUILD_DIR"
    cd "$BUILD_DIR"
else
    echo "错误：构建目录 $BUILD_DIR 不存在！"
    exit 1
fi

# 检查 genTelechat 可执行文件是否存在
if [ -x "$GEN_TELECHAT_EXECUTABLE" ]; then
    echo "运行 genTelechat..."
    "$GEN_TELECHAT_EXECUTABLE"
else
    echo "错误：文件 $GEN_TELECHAT_EXECUTABLE 不存在或不可执行！"
    exit 1
fi

# 检查 Telechat 可执行文件是否存在
if [ -x "$TELECHAT_EXECUTABLE" ]; then
    echo "运行 Telechat..."
    "$TELECHAT_EXECUTABLE"
else
    echo "错误：文件 $TELECHAT_EXECUTABLE 不存在或不可执行！"
    exit 1
fi
