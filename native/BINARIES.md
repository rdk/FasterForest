# Pre-compiled native binaries

The committed shared libraries under `src/main/resources/native/` are built from the C sources in this
directory (`native/src/*.c`, `native/CMakeLists.txt`) — see [../BUILDING.md](../BUILDING.md). They are
loaded at runtime by the `NativePanama*` forests (Java 22+); the library is optional — the JVM forests
work without it.

## Checksums & provenance

Verify a binary with `sha256sum <file>` and compare to the table. **Regenerate this table whenever a
binary is rebuilt** (`sha256sum`, the new size, and the commit that rebuilt it).

| File | size (bytes) | sha256 | md5 | last rebuilt (commit, date) |
|---|---|---|---|---|
| `src/main/resources/native/linux-x86_64/fasterforest.so` | 20696 | `8d5dde9db26d7f478f8a05fb5c68db3579b8288201dac8946f0c96df65a6d0a1` | `d2fff4c3972be012294641cab96d41f1` | `c1a8ece` 2026-02-23 (rebuilt with -O3) |
| `src/main/resources/native/windows-x86_64/fasterforest.dll` | 13824 | `994e8a093a5e76a87f36d57904a5ffa09c2437df7a098bfebeae5736ebf915da` | `7f70988db532ee59af81032135a19cb0` | `4dfa556` 2026-02-23 |

## Build

- **Toolchain:** CMake 3.16+, GCC (Linux) / MSVC (Windows), Release / `-O3`.
- **SIMD:** scalar and AVX2 kernels are both compiled (`predict_scalar*.c`, `predict_avx2*.c`); the AVX2
  path is selected at **runtime** via CPU feature detection, so the binaries run on non-AVX2 CPUs.
- **One-shot rebuild:** `./build-linux.sh` / `./build-windows.sh` (build native lib + copy into
  `src/main/resources/native/...` + build the JAR). Manual CMake steps are in [../BUILDING.md](../BUILDING.md).

After rebuilding, update the checksum row above in the same commit.
