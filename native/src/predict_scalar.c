#include "forest.h"
#include "../include/fasterforest.h"
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Runtime SIMD dispatch                                                     */
/* ========================================================================= */

static int detect_simd_level(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    /* Check for AVX2 via CPUID */
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

    /* AVX2 is bit 5 of EBX from CPUID leaf 7 */
    if (info[1] & (1 << 5)) return 2;
#endif
    return 0; /* scalar fallback */
}

static ff_batch_fn g_batch_fn = NULL;

/* Eagerly initialize SIMD dispatch at library load time (before any exported
   function can be called).  Eliminates the TOCTOU race on g_batch_fn. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void init_dispatch(void) {
    int level = detect_simd_level();
#if defined(__x86_64__) || defined(_M_X64)
    if (level >= 2) {
        g_batch_fn = ff_predict_batch_avx2;
        return;
    }
#endif
    (void)level;
    g_batch_fn = ff_predict_batch_scalar;
}

#if defined(_MSC_VER)
/* MSVC: run init_dispatch via DllMain */
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL; (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_ATTACH) init_dispatch();
    return TRUE;
}
#endif

/* ========================================================================= */
/* Forest create / destroy                                                   */
/* ========================================================================= */

FF_API FfForest* ff_forest_create(
    int32_t num_trees, int32_t num_attributes,
    int32_t total_nodes, int32_t total_leaves,
    const int32_t* tree_roots, const int32_t* child_left,
    const int32_t* child_right, const int32_t* attr_index,
    const double* split_point, const double* score)
{
    if (num_trees <= 0 || num_attributes <= 0) return NULL;

    FfForest* f = (FfForest*)malloc(sizeof(FfForest));
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

    /* init_dispatch() runs at library load via constructor/__attribute__. */

    return f;
}

FF_API void ff_forest_destroy(FfForest* forest) {
    free(forest);
}

/* ========================================================================= */
/* Scalar prediction                                                         */
/* ========================================================================= */

FF_API double ff_predict(const FfForest* f, const double* inst) {
    const int32_t* cl = f->child_left;
    const int32_t* cr = f->child_right;
    const int32_t* ai = f->attr_index;
    const double*  sp = f->split_point;
    const double*  sc = f->score;

    double sum = 0.0;
    for (int32_t t = 0; t < f->num_trees; ++t) {
        int32_t node = f->tree_roots[t];
        for (;;) {
            int32_t left  = cl[node];
            int32_t right = cr[node];
            node = (inst[ai[node]] < sp[node]) ? left : right;
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

void ff_predict_batch_scalar(
    const FfForest* f,
    const double* instances, int32_t n,
    double* out)
{
    const int32_t na = f->num_attributes;
    const int32_t* cl = f->child_left;
    const int32_t* cr = f->child_right;
    const int32_t* ai = f->attr_index;
    const double*  sp = f->split_point;
    const double*  sc = f->score;

    memset(out, 0, (size_t)n * sizeof(double));

    for (int32_t t = 0; t < f->num_trees; ++t) {
        const int32_t root = f->tree_roots[t];
        for (int32_t i = 0; i < n; ++i) {
            const double* inst = instances + (int64_t)i * na;
            int32_t node = root;
            for (;;) {
                int32_t left  = cl[node];
                int32_t right = cr[node];
                node = (inst[ai[node]] < sp[node]) ? left : right;
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

FF_API void ff_predict_batch(
    const FfForest* f,
    const double* instances, int32_t n,
    double* out)
{
    g_batch_fn(f, instances, n, out);
}

FF_API int ff_simd_level(void) {
    if (g_batch_fn == ff_predict_batch_avx2) return 2;
    return 0;
}
