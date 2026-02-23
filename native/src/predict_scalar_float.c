#include "forest.h"
#include "../include/fasterforest.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Runtime SIMD dispatch for float forest                                    */
/* ========================================================================= */

static ff_float_batch_fn g_float_batch_fn = NULL;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void init_float_dispatch(void) {
#if defined(__x86_64__) || defined(_M_X64)
    /* Reuse SIMD level from the double variant (same CPU). Check AVX2 via CPUID. */
    int info[4] = {0};
  #if defined(_MSC_VER)
    __cpuidex(info, 7, 0);
  #elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(7), "c"(0)
    );
  #endif
    if (info[1] & (1 << 5)) {
        g_float_batch_fn = ff_float_predict_batch_avx2;
        return;
    }
#endif
    g_float_batch_fn = ff_float_predict_batch_scalar;
}

#if defined(_MSC_VER)
/* MSVC: run via DllMain in predict_scalar.c (already handles DLL_PROCESS_ATTACH).
   For float dispatch we use a CRT initializer instead. */
#pragma section(".CRT$XCU", read)
static void __cdecl init_float_dispatch_msvc(void) { init_float_dispatch(); }
__declspec(allocate(".CRT$XCU")) static void (__cdecl *p_init_float)(void) = init_float_dispatch_msvc;
#endif

/* ========================================================================= */
/* Forest create / destroy                                                   */
/* ========================================================================= */

FF_API FfFloatForest* ff_float_forest_create(
    int32_t num_trees, int32_t num_attributes,
    int32_t total_nodes, int32_t total_leaves,
    const int32_t* tree_roots, const int32_t* child_left,
    const int32_t* child_right, const int32_t* attr_index,
    const float* split_point, const float* score)
{
    if (num_trees <= 0 || num_attributes <= 0) return NULL;

    FfFloatForest* f = (FfFloatForest*)malloc(sizeof(FfFloatForest));
    if (!f) return NULL;

    f->num_trees      = num_trees;
    f->num_attributes = num_attributes;
    f->total_nodes    = total_nodes;
    f->total_leaves   = total_leaves;
    f->tree_roots     = tree_roots;
    f->child_left     = child_left;
    f->child_right    = child_right;
    f->attr_index     = attr_index;
    f->split_point    = split_point;
    f->score          = score;
    f->inv_num_trees  = 1.0 / num_trees;

    return f;
}

FF_API void ff_float_forest_destroy(FfFloatForest* forest) {
    free(forest);
}

/* ========================================================================= */
/* Scalar prediction                                                         */
/* ========================================================================= */

FF_API double ff_float_predict(const FfFloatForest* f, const double* inst) {
    const int32_t* cl = f->child_left;
    const int32_t* cr = f->child_right;
    const int32_t* ai = f->attr_index;
    const float*   sp = f->split_point;
    const float*   sc = f->score;

    double sum = 0.0;
    for (int32_t t = 0; t < f->num_trees; ++t) {
        int32_t node = f->tree_roots[t];
        for (;;) {
            int32_t left  = cl[node];
            int32_t right = cr[node];
            node = ((float)inst[ai[node]] < sp[node]) ? left : right;
            if (node < 0) {
                sum += sc[-node];
                break;
            }
        }
    }
    return sum * f->inv_num_trees;
}

/* ========================================================================= */
/* Scalar batch prediction                                                   */
/* ========================================================================= */

void ff_float_predict_batch_scalar(
    const FfFloatForest* f,
    const double* instances, int32_t n,
    double* out)
{
    const int32_t na = f->num_attributes;
    const int32_t* cl = f->child_left;
    const int32_t* cr = f->child_right;
    const int32_t* ai = f->attr_index;
    const float*   sp = f->split_point;
    const float*   sc = f->score;

    memset(out, 0, (size_t)n * sizeof(double));

    for (int32_t t = 0; t < f->num_trees; ++t) {
        const int32_t root = f->tree_roots[t];
        for (int32_t i = 0; i < n; ++i) {
            const double* inst = instances + (int64_t)i * na;
            int32_t node = root;
            for (;;) {
                int32_t left  = cl[node];
                int32_t right = cr[node];
                node = ((float)inst[ai[node]] < sp[node]) ? left : right;
                if (node < 0) {
                    out[i] += sc[-node];
                    break;
                }
            }
        }
    }

    const double inv = f->inv_num_trees;
    for (int32_t i = 0; i < n; ++i) {
        out[i] *= inv;
    }
}

/* ========================================================================= */
/* Dispatched batch prediction                                               */
/* ========================================================================= */

FF_API void ff_float_predict_batch(
    const FfFloatForest* f,
    const double* instances, int32_t n,
    double* out)
{
    g_float_batch_fn(f, instances, n, out);
}

FF_API void ff_float_predict_batch_scalar_only(
    const FfFloatForest* f,
    const double* instances, int32_t n,
    double* out)
{
    ff_float_predict_batch_scalar(f, instances, n, out);
}
