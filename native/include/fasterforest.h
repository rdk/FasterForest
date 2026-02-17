#ifndef FASTERFOREST_H
#define FASTERFOREST_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef FF_BUILD_DLL
    #define FF_API __declspec(dllexport)
  #else
    #define FF_API __declspec(dllimport)
  #endif
#else
  #define FF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque forest handle. Holds pointers to the forest data arrays.
 * Does NOT own the array memory - the caller (Java Arena) owns it.
 */
typedef struct FfForest FfForest;

/**
 * Create a forest handle from pre-allocated arrays.
 * All pointers must remain valid for the lifetime of the returned handle.
 * The handle is a small malloc'd struct.
 *
 * Returns NULL on failure (invalid arguments).
 */
FF_API FfForest* ff_forest_create(
    int32_t  num_trees,
    int32_t  num_attributes,
    int32_t  total_nodes,
    int32_t  total_leaves,
    const int32_t* tree_roots,
    const int32_t* child_left,
    const int32_t* child_right,
    const int32_t* attr_index,
    const double*  split_point,
    const double*  score
);

/**
 * Free the forest handle (NOT the data arrays).
 */
FF_API void ff_forest_destroy(FfForest* forest);

/**
 * Predict a single instance.
 * instance_attrs: double[num_attributes]
 * Returns predicted probability of class 1 (0.0 to 1.0).
 */
FF_API double ff_predict(
    const FfForest* forest,
    const double*   instance_attrs
);

/**
 * Predict a batch of instances.
 * instances: contiguous row-major double[n * num_attributes]
 * out:       output double[n], caller-allocated
 * n:         number of instances
 */
FF_API void ff_predict_batch(
    const FfForest* forest,
    const double*   instances,
    int32_t         n,
    double*         out
);

/**
 * Query which SIMD level is active.
 * Returns: 0=scalar, 2=AVX2, 3=AVX-512
 */
FF_API int ff_simd_level(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTERFOREST_H */
