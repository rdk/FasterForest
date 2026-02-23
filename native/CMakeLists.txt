cmake_minimum_required(VERSION 3.16)
project(fasterforest C)

set(CMAKE_C_STANDARD 11)

# --- Scalar source (baseline, always built) ---
add_library(fasterforest SHARED
    src/predict_scalar.c
    src/predict_scalar_float.c
)

target_include_directories(fasterforest PUBLIC include PRIVATE src)
target_compile_definitions(fasterforest PRIVATE FF_BUILD_DLL)

if(MSVC)
    target_compile_options(fasterforest PRIVATE /O2 /W4 /fp:precise)
else()
    target_compile_options(fasterforest PRIVATE -O3 -Wall -Wextra -fvisibility=hidden -ffp-contract=off)
endif()

# --- AVX2 source (compiled with AVX2 flags, x86_64 only) ---
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(fasterforest_avx2 OBJECT
        src/predict_avx2.c
        src/predict_avx2_float.c
    )
    target_include_directories(fasterforest_avx2 PRIVATE include src)

    if(MSVC)
        target_compile_options(fasterforest_avx2 PRIVATE /O2 /arch:AVX2 /fp:precise)
        target_compile_definitions(fasterforest_avx2 PRIVATE FF_BUILD_DLL)
    else()
        target_compile_options(fasterforest_avx2 PRIVATE -O3 -mavx2 -mfma -ffp-contract=off -fvisibility=hidden)
    endif()

    target_link_libraries(fasterforest PRIVATE fasterforest_avx2)
endif()

# Output naming
set_target_properties(fasterforest PROPERTIES
    OUTPUT_NAME "fasterforest"
)
