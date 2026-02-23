#ifndef FOREST_INTERNAL_H
#define FOREST_INTERNAL_H

#include <stdint.h>

typedef struct FfForest {
    int32_t  num_trees;
    int32_t  num_attributes;
    int32_t  total_nodes;
    int32_t  total_leaves;
    const int32_t* tree_roots;
    const int32_t* child_left;
    const int32_t* child_right;
    const int32_t* attr_index;
    const double*  split_point;
    const double*  score;
    double   inv_num_trees;
} FfForest;

/* Batch prediction function pointer type for runtime dispatch */
typedef void (*ff_batch_fn)(const FfForest*, const double*, int32_t, double*);

/* Scalar batch implementation (always available) */
void ff_predict_batch_scalar(const FfForest* f, const double* instances, int32_t n, double* out);

/* AVX2 batch implementation (compiled separately with -mavx2) */
void ff_predict_batch_avx2(const FfForest* f, const double* instances, int32_t n, double* out);

/* ========================================================================= */
/* Float-precision forest                                                    */
/* ========================================================================= */

typedef struct FfFloatForest {
    int32_t  num_trees;
    int32_t  num_attributes;
    int32_t  total_nodes;
    int32_t  total_leaves;
    const int32_t* tree_roots;
    const int32_t* child_left;
    const int32_t* child_right;
    const int32_t* attr_index;
    const float*   split_point;
    const float*   score;
    double   inv_num_trees;  /* double precision for final multiply */
} FfFloatForest;

/* Batch prediction function pointer type for float forest */
typedef void (*ff_float_batch_fn)(const FfFloatForest*, const double*, int32_t, double*);

/* Scalar batch implementation (always available) */
void ff_float_predict_batch_scalar(const FfFloatForest* f, const double* instances, int32_t n, double* out);

/* AVX2 batch implementation (compiled separately with -mavx2) */
void ff_float_predict_batch_avx2(const FfFloatForest* f, const double* instances, int32_t n, double* out);

#endif /* FOREST_INTERNAL_H */
