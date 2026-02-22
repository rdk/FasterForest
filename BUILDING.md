# Building FasterForest

## Prerequisites

- **JDK 22+** (required for Panama FFM native access)
- **Gradle** (wrapper included, no install needed)

For native library builds:
- **CMake 3.16+**
- **GCC** (Linux) or **MSVC** (Windows)

## Quick Build (Java only)

If you don't need to recompile the native C library (pre-compiled binaries are
already included in `src/main/resources/native/`):

```bash
./gradlew clean assemble
```

The JAR is output to `build/libs/FasterForest-<version>.jar`.

## Building with Native Library

The native library provides C-based forest prediction via Panama FFM.
Pre-compiled binaries for Linux and Windows (x86_64) are committed to the repo,
so you only need to rebuild them if you change the C source code.

### Linux x86_64

```bash
./build-linux.sh
```

This script:
1. Builds the native `libfasterforest.so` via CMake
2. Copies it to `src/main/resources/native/linux-x86_64/fasterforest.so`
3. Builds the JAR with `./gradlew clean assemble`

Or manually:
```bash
cd native
mkdir -p build && cd build
cmake ..
make
cp libfasterforest.so ../../src/main/resources/native/linux-x86_64/fasterforest.so
cd ../..
./gradlew clean assemble
```

### Windows x86_64

With MSVC (Visual Studio):
```bash
cd native
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
copy Release\fasterforest.dll ..\..\src\main\resources\native\windows-x86_64\fasterforest.dll
cd ..\..
gradlew clean assemble
```

With MinGW:
```bash
cd native
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
copy libfasterforest.dll ..\..\src\main\resources\native\windows-x86_64\fasterforest.dll
cd ..\..
gradlew clean assemble
```

### Cross-platform JAR

To build a JAR containing native libraries for both platforms:

1. Build the native library on each platform (or cross-compile)
2. Place the binaries in the resource tree:
   - `src/main/resources/native/linux-x86_64/fasterforest.so`
   - `src/main/resources/native/windows-x86_64/fasterforest.dll`
3. Run `./gradlew clean assemble`

The JAR bundles all native libraries found in the resource tree.
At runtime, `NativeLoader` extracts the correct one for the current OS/arch.

## Notes

- AVX2 SIMD code is compiled only on x86_64. Runtime CPUID detection
  falls back to scalar if AVX2 is not available on the target machine.
- The native library is optional. If loading fails, FasterForest falls back
  to pure Java prediction automatically.
- For development, you can point to a local native library with:
  `-Dfasterforest.native.path=/path/to/libfasterforest.so`
