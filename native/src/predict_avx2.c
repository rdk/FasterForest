#include "forest.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

/**
 * AVX2 batch prediction - processes 4 instances simultaneously through each tree.
 *
 * Uses AVX2 to compare 4 split attributes at once, then selects children per lane.
 * Since tree traversal paths diverge (different instances hit leaves at different depths),
 * we use a mask to skip completed lanes.
 */
void ff_predict_batch_avx2(
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

    const int32_t n4 = n & ~3; /* round down to multiple of 4 */

    for (int32_t t = 0; t < f->num_trees; ++t) {
        const int32_t root = f->tree_roots[t];

        /* AVX2 loop: 4 instances at a time */
        for (int32_t i = 0; i < n4; i += 4) {
            const double* i0 = instances + (int64_t)(i + 0) * na;
            const double* i1 = instances + (int64_t)(i + 1) * na;
            const double* i2 = instances + (int64_t)(i + 2) * na;
            const double* i3 = instances + (int64_t)(i + 3) * na;

            int32_t n0 = root, n1 = root, n2 = root, n3 = root;

            for (;;) {
                int active = (n0 >= 0) | ((n1 >= 0) << 1) | ((n2 >= 0) << 2) | ((n3 >= 0) << 3);
                if (!active) break;

                /* Gather attribute values for 4 instances */
                __m256d vals = _mm256_set_pd(
                    n3 >= 0 ? i3[ai[n3]] : 0.0,
                    n2 >= 0 ? i2[ai[n2]] : 0.0,
                    n1 >= 0 ? i1[ai[n1]] : 0.0,
                    n0 >= 0 ? i0[ai[n0]] : 0.0
                );

                /* Gather split points */
                __m256d sps = _mm256_set_pd(
                    n3 >= 0 ? sp[n3] : 0.0,
                    n2 >= 0 ? sp[n2] : 0.0,
                    n1 >= 0 ? sp[n1] : 0.0,
                    n0 >= 0 ? sp[n0] : 0.0
                );

                /* Compare: vals < sps → mask bit set if true (go left) */
                __m256d cmp = _mm256_cmp_pd(vals, sps, _CMP_LT_OQ);
                int mask = _mm256_movemask_pd(cmp);

                /* Select child per lane based on comparison result */
                if (n0 >= 0) n0 = (mask & 1) ? cl[n0] : cr[n0];
                if (n1 >= 0) n1 = (mask & 2) ? cl[n1] : cr[n1];
                if (n2 >= 0) n2 = (mask & 4) ? cl[n2] : cr[n2];
                if (n3 >= 0) n3 = (mask & 8) ? cl[n3] : cr[n3];
            }

            out[i + 0] += sc[-n0];
            out[i + 1] += sc[-n1];
            out[i + 2] += sc[-n2];
            out[i + 3] += sc[-n3];
        }

        /* Scalar tail for remaining instances */
        for (int32_t i = n4; i < n; ++i) {
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

#else
/* Non-x86: AVX2 not available, fall back to scalar */
void ff_predict_batch_avx2(
    const FfForest* f,
    const double* instances, int32_t n,
    double* out)
{
    ff_predict_batch_scalar(f, instances, n, out);
}
#endif
