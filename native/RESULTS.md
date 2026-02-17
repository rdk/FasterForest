# Native Panama FFM - Implementation Results

## Summary

Native C prediction via Panama FFM (Java 22+) is **33% slower** than the best pure-Java
implementation. The C prediction code itself (cmov + AVX2) is likely faster, but the cost
of copying `double[][]` to contiguous off-heap memory on every call negates the gains.

## Benchmark Results (2026-02-17)

Environment: Windows 10, JDK 22, AVX2 (SIMD level 2), VS2022 `/O2 /fp:precise`
Dataset: 6950 instances, 29 attributes, 100 trees, max depth 25
Config: warmup=3, measured=10, iters=400 (2,780,000 predictions/round)

| #  | Forest                | Mean ms | Std ms | Pred/sec | vs Best |
|----|-----------------------|---------|--------|----------|---------|
|  1 | SeparateArraysBfs     |  5421.5 |  126.9 |  512,773 |       - |
|  2 | ContiguousDfs         |  5488.3 |  239.8 |  506,532 |     -1% |
|  3 | Flat                  |  5590.8 |  159.8 |  497,245 |     -3% |
|  4 | Original (FF)         |  5592.2 |  148.2 |  497,121 |     -3% |
|  5 | InterleavedBfsDouble  |  5705.6 |   58.5 |  487,241 |     -5% |
|  6 | InterleavedBfs        |  6011.5 |  175.7 |  462,447 |    -10% |
|  7 | ContiguousBfsDouble   |  6083.2 | 1520.7 |  456,996 |    -11% |
|  8 | LegacyFlat            |  6123.7 |  170.1 |  453,974 |    -12% |
|  9 | ShortLegacy           |  6723.6 | 1682.8 |  413,469 |    -19% |
| 10 | **NativePanama**      |  7324.4 |  268.9 |  379,553 |  **-33%** |
| 11 | IlpDfs                |  7810.6 |  108.3 |  355,927 |    -35% |
| 12 | BranchlessBfs         |  8998.0 |  223.4 |  308,958 |    -43% |

## Why Native is Slower

The bottleneck is **not** the C prediction code -- it's the data marshalling:

1. **`double[][]` to contiguous copy** (~1.6 MB per batch call): Java's `double[][]` is an
   array of pointers to separate row arrays scattered across the heap. The C code expects a
   single contiguous row-major buffer. Every `predictForBatch()` call copies all rows into a
   pre-allocated off-heap `MemorySegment`. Java implementations access `double[][]` directly
   with zero copy cost.

2. **Result copy back**: After native prediction, the `double[]` result is copied from
   off-heap memory back to a Java array.

3. **Pre-allocated buffers reduced variance but not throughput**: Reusing `instanceBuffer`
   and `outputBuffer` eliminated per-call Arena allocation overhead (std dropped from
   1532ms to 269ms) but the memcpy itself is irreducible at ~1.6 MB per call.

## What Was Implemented

- **Phase 1 (Scalar)**: `predict_scalar.c` -- cmov-friendly traversal loop, no bounds checks
- **Phase 2 (AVX2)**: `predict_avx2.c` -- processes 4 instances through same tree simultaneously
  using `_mm256_cmp_pd` + `_mm256_movemask_pd`, with active mask for completed lanes
- **Runtime dispatch**: CPUID-based AVX2 detection, function pointer dispatch via `g_batch_fn`
- **Correctness**: Bit-identical predictions verified against Java `ContiguousDfsForest` using
  `Double.doubleToRawLongBits()` comparison (both single and batch)

## Ideas to Make Native Competitive

### 1. Pre-flattened off-heap data (most impactful)
Keep instances in contiguous off-heap memory from the start. Provide
`predictForBatchContiguous(MemorySegment data, int n)` that skips the copy entirely.
Requires callers to arrange data in off-heap memory -- API change upstream (P2Rank).

### 2. Batch size threshold
Only use native for very large batches where AVX2 speedup outweighs copy cost.
For small batches, fall back to Java ContiguousDfsForest automatically.

### 3. Profile native code in isolation
Use VTune/perf to measure prediction-only time (excluding copy). If prediction itself is
2-3x faster than Java, the copy overhead is confirmed as the sole bottleneck and the path
forward is clear: eliminate the copy.

### 4. Streaming prediction
Interleave copy and predict in chunks. Overlaps memcpy with computation. Small expected
gain since both operations are memory-bandwidth bound.

### 5. Memory-mapped instance data
For offline/batch scoring, load instances directly from a binary file into off-heap memory
via `Arena.mapFile()`. Zero-copy path from disk to native prediction.

## Conclusion

Native C + SIMD is not a viable optimization for this use case **given the current Java API
contract** (`double[][]` input). The data copy overhead dominates. The approach would become
viable if the upstream data pipeline (P2Rank) could provide instances in contiguous off-heap
memory, bypassing the copy entirely.

For the current API, pure Java `SeparateArraysBfs` or `ContiguousDfs` remain the fastest
implementations.
