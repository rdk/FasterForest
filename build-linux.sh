#!/bin/bash

# Build FasterForest native library and JAR on Linux x86_64.
# Requires: JDK 22+, CMake 3.16+, GCC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATIVE_DIR="$SCRIPT_DIR/native"
BUILD_DIR="$NATIVE_DIR/build"
RESOURCE_DIR="$SCRIPT_DIR/src/main/resources/native/linux-x86_64"

echo "=== Building native library ==="

mkdir -p "$BUILD_DIR"
cmake -S "$NATIVE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --clean-first

echo "=== Installing native library to resources ==="

mkdir -p "$RESOURCE_DIR"
cp "$BUILD_DIR/libfasterforest.so" "$RESOURCE_DIR/fasterforest.so"
echo "Copied to $RESOURCE_DIR/fasterforest.so"

echo "=== Building JAR ==="

"$SCRIPT_DIR/gradlew" -p "$SCRIPT_DIR" clean assemble

echo "=== Done ==="
ls -lh "$SCRIPT_DIR"/build/libs/*.jar
