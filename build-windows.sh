#!/bin/bash

# Build FasterForest native library and JAR on Windows x86_64 (Git Bash / MSYS2).
# Requires: JDK 22+, CMake 3.16+, MSVC or MinGW

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATIVE_DIR="$SCRIPT_DIR/native"
BUILD_DIR="$NATIVE_DIR/build"
RESOURCE_DIR="$SCRIPT_DIR/src/main/resources/native/windows-x86_64"

echo "=== Building native library ==="

mkdir -p "$BUILD_DIR"
cmake -S "$NATIVE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release --clean-first

echo "=== Installing native library to resources ==="

mkdir -p "$RESOURCE_DIR"

# Find the built DLL (MSVC puts it under Release/, MinGW in the build root)
if [ -f "$BUILD_DIR/Release/fasterforest.dll" ]; then
    cp "$BUILD_DIR/Release/fasterforest.dll" "$RESOURCE_DIR/fasterforest.dll"
elif [ -f "$BUILD_DIR/libfasterforest.dll" ]; then
    cp "$BUILD_DIR/libfasterforest.dll" "$RESOURCE_DIR/fasterforest.dll"
else
    echo "ERROR: Could not find built DLL in $BUILD_DIR" >&2
    exit 1
fi
echo "Copied to $RESOURCE_DIR/fasterforest.dll"

echo "=== Building JAR ==="

"$SCRIPT_DIR/gradlew.bat" -p "$SCRIPT_DIR" clean assemble

echo "=== Done ==="
ls -lh "$SCRIPT_DIR"/build/libs/*.jar
