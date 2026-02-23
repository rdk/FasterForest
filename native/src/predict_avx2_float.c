#include "forest.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

/**
 * AVX2 batch prediction for float forest - processes 8 instances simultaneously.
 *
 * Uses 256-bit AVX2 to compare 8 float split points at once.
 * Instance attributes (doubles) are cast to float before comparison.
 * Score accumulation stays in double for precision.
 */
void ff_float_predict_batch_avx2(
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

    const int32_t n8 = n & ~7; /* round down to multiple of 8 */

    for (int32_t t = 0; t < f->num_trees; ++t) {
        const int32_t root = f->tree_roots[t];

        /* AVX2 loop: 8 instances at a time */
        for (int32_t i = 0; i < n8; i += 8) {
            const double* i0 = instances + (int64_t)(i + 0) * na;
            const double* i1 = instances + (int64_t)(i + 1) * na;
            const double* i2 = instances + (int64_t)(i + 2) * na;
            const double* i3 = instances + (int64_t)(i + 3) * na;
            const double* i4 = instances + (int64_t)(i + 4) * na;
            const double* i5 = instances + (int64_t)(i + 5) * na;
            const double* i6 = instances + (int64_t)(i + 6) * na;
            const double* i7 = instances + (int64_t)(i + 7) * na;

            int32_t n0 = root, n1 = root, n2 = root, n3 = root;
            int32_t n4 = root, n5 = root, n6 = root, n7 = root;

            for (;;) {
                int active = (n0 >= 0)       | ((n1 >= 0) << 1) |
                             ((n2 >= 0) << 2) | ((n3 >= 0) << 3) |
                             ((n4 >= 0) << 4) | ((n5 >= 0) << 5) |
                             ((n6 >= 0) << 6) | ((n7 >= 0) << 7);
                if (!active) break;

                /* Gather attribute values (double→float) with guards for inactive lanes */
                __m256 vals = _mm256_set_ps(
                    n7 >= 0 ? (float)i7[ai[n7]] : 0.0f,
                    n6 >= 0 ? (float)i6[ai[n6]] : 0.0f,
                    n5 >= 0 ? (float)i5[ai[n5]] : 0.0f,
                    n4 >= 0 ? (float)i4[ai[n4]] : 0.0f,
                    n3 >= 0 ? (float)i3[ai[n3]] : 0.0f,
                    n2 >= 0 ? (float)i2[ai[n2]] : 0.0f,
                    n1 >= 0 ? (float)i1[ai[n1]] : 0.0f,
                    n0 >= 0 ? (float)i0[ai[n0]] : 0.0f
                );

                /* Gather float split points with guards */
                __m256 sps = _mm256_set_ps(
                    n7 >= 0 ? sp[n7] : 0.0f,
                    n6 >= 0 ? sp[n6] : 0.0f,
                    n5 >= 0 ? sp[n5] : 0.0f,
                    n4 >= 0 ? sp[n4] : 0.0f,
                    n3 >= 0 ? sp[n3] : 0.0f,
                    n2 >= 0 ? sp[n2] : 0.0f,
                    n1 >= 0 ? sp[n1] : 0.0f,
                    n0 >= 0 ? sp[n0] : 0.0f
                );

                /* Compare: vals < sps → mask bit set if true (go left) */
                __m256 cmp = _mm256_cmp_ps(vals, sps, _CMP_LT_OQ);
                int mask = _mm256_movemask_ps(cmp);

                /* Select child per lane based on comparison result */
                if (n0 >= 0) n0 = (mask &   1) ? cl[n0] : cr[n0];
                if (n1 >= 0) n1 = (mask &   2) ? cl[n1] : cr[n1];
                if (n2 >= 0) n2 = (mask &   4) ? cl[n2] : cr[n2];
                if (n3 >= 0) n3 = (mask &   8) ? cl[n3] : cr[n3];
                if (n4 >= 0) n4 = (mask &  16) ? cl[n4] : cr[n4];
                if (n5 >= 0) n5 = (mask &  32) ? cl[n5] : cr[n5];
                if (n6 >= 0) n6 = (mask &  64) ? cl[n6] : cr[n6];
                if (n7 >= 0) n7 = (mask & 128) ? cl[n7] : cr[n7];
            }

            out[i + 0] += sc[-n0];
            out[i + 1] += sc[-n1];
            out[i + 2] += sc[-n2];
            out[i + 3] += sc[-n3];
            out[i + 4] += sc[-n4];
            out[i + 5] += sc[-n5];
            out[i + 6] += sc[-n6];
            out[i + 7] += sc[-n7];
        }

        /* Scalar tail for remaining instances */
        for (int32_t i = n8; i < n; ++i) {
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

#else
/* Non-x86: AVX2 not available, fall back to scalar */
void ff_float_predict_batch_avx2(
    const FfFloatForest* f,
    const double* instances, int32_t n,
    double* out)
{
    ff_float_predict_batch_scalar(f, instances, n, out);
}
#endif
