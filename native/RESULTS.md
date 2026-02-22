# Native Panama FFM - Implementation Results

## Summary

Native C prediction via Panama FFM (Java 22+) is slower than the best pure-Java
implementation on both Windows (**34% slower**) and Linux (**25% slower**). A zero-copy
path (`predictForBatchContiguous`) showed negligible improvement on both platforms,
disproving the hypothesis that data marshalling was the bottleneck. The native C prediction
loop itself is fundamentally slower than Java's JIT-compiled code for this workload.

## Benchmark Results (2026-02-17, run 2 - with zero-copy)

Environment: Windows 10 (Ryzen 5 9600X), JDK 22, AVX2 (SIMD level 2), VS2022 `/O2 /fp:precise`
Dataset: 6950 instances, 29 attributes, 100 trees, max depth 25
Config: warmup=3, measured=10, iters=400 (2,780,000 predictions/round)

| #  | Forest                | Mean ms | Std ms | Pred/sec | vs Best |
|----|-----------------------|---------|--------|----------|---------|
|  1 | ContiguousDfs         |  5372.6 |   53.7 |  517,440 |       - |
|  2 | Original (FF)         |  5365.0 |  212.0 |  518,173 |     ~0% |
|  3 | SeparateArraysBfs     |  5483.5 |  103.1 |  506,975 |     -2% |
|  4 | Flat                  |  5559.0 |   95.9 |  500,090 |     -3% |
|  5 | ContiguousBfsDouble   |  5689.1 |  153.8 |  488,654 |     -6% |
|  6 | InterleavedBfsDouble  |  5818.6 |  211.3 |  477,778 |     -8% |
|  7 | InterleavedBfs        |  6575.8 | 1487.9 |  422,762 |    -18% |
|  8 | ShortLegacy           |  6537.6 |   88.8 |  425,233 |    -18% |
|  9 | LegacyFlat            |  6902.9 | 2354.3 |  402,729 |    -22% |
| 10 | **NativePanama0copy** |  7201.6 |  174.7 |  386,025 |  **-34%** |
| 11 | **NativePanama**      |  7256.6 |  209.8 |  383,100 |  **-35%** |
| 12 | IlpDfs                |  7898.7 |  160.0 |  351,957 |    -38% |
| 13 | BranchlessBfs         |  9440.0 | 1360.4 |  294,492 |    -43% |

### Key finding: zero-copy is NOT faster

NativePanama0copy (pre-flattened off-heap data, no per-call copy): **7201.6 ms**
NativePanama (copy double[][] to off-heap every call):             **7256.6 ms**
Difference: **< 1%** (55 ms, within noise)

This disproves the hypothesis that data marshalling overhead was responsible for the
native path being slower. The C prediction loop itself is the bottleneck.

## Benchmark Results (2026-02-22, Linux)

Environment: Linux 6.14 (Ryzen 5 9600X), GraalVM JDK 24, AVX2 (SIMD level 2), GCC 14.2 `-O2 -ffp-contract=off`
Dataset: 6950 instances, 29 attributes, 100 trees, max depth 25
Config: warmup=3, measured=10, iters=400 (2,780,000 predictions/round)

| #  | Forest                | Mean ms | Std ms | Pred/sec | vs Best |
|----|-----------------------|---------|--------|----------|---------|
|  1 | ContiguousDfs         |  5036.1 |   27.5 |  552,014 |       - |
|  2 | SeparateArraysBfs     |  5062.4 |   38.5 |  549,147 |     -1% |
|  3 | LegacyFlat            |  5124.0 |   27.4 |  542,545 |     -2% |
|  4 | Flat                  |  5124.5 |   11.3 |  542,492 |     -2% |
|  5 | ShortLegacy           |  5148.7 |   25.3 |  539,942 |     -2% |
|  6 | Original (FF)         |  5218.1 |   16.2 |  532,761 |     -4% |
|  7 | ContiguousBfsDouble   |  5356.2 |    7.4 |  519,025 |     -6% |
|  8 | InterleavedBfsDouble  |  5514.8 |   24.9 |  504,098 |     -9% |
|  9 | InterleavedBfs        |  5862.3 |   43.9 |  474,217 |    -14% |
| 10 | **NativePanama0copy** |  6313.1 |   24.0 |  440,354 |  **-25%** |
| 11 | **NativePanama**      |  6423.8 |   17.1 |  432,766 |  **-28%** |
| 12 | IlpDfs                |  6775.9 |   26.9 |  410,278 |    -26% |
| 13 | BranchlessBfs         |  8221.9 |    4.8 |  338,121 |    -39% |

### Cross-platform observations

- **Native gap is smaller on Linux** (25-28%) than Windows (34-35%) — GCC produces
  better code than MSVC for this workload
- **Java is faster on Linux** too: ContiguousDfs 5036 ms (Linux) vs 5373 ms (Windows),
  a ~6% improvement likely due to GraalVM JIT vs HotSpot JDK 22
- **Standard deviations are much lower on Linux** — more stable measurements across
  the board (e.g. InterleavedBfs std 44 ms vs 1488 ms on Windows)
- **LegacyFlat and ShortLegacy rank much higher on Linux** — the large std deviations
  that hurt them on Windows disappear here
- **Zero-copy gap slightly larger on Linux**: 110 ms (1.7%) vs 55 ms (<1%) — still
  small, confirming that data copy is not the bottleneck on either platform
- The overall conclusion holds: **pure Java beats native C+AVX2 on both platforms**

## Why Native is Slower (updated analysis)

The original hypothesis was that `double[][]` → off-heap copy overhead caused the slowdown.
This was disproven by `predictForBatchContiguous` which eliminates all data copying but
shows negligible improvement.

The actual bottleneck is the **native prediction loop itself**:

1. **Panama FFM downcall overhead**: Each `ff_predict_batch` call crosses the Java→native
   boundary via Panama's `MethodHandle.invokeExact()`. While individually small (~20-30ns),
   this is called 400 times per measured round.

2. **JIT superiority for this workload**: HotSpot's C2 JIT compiler produces highly optimized
   code for the simple tree traversal loop. It benefits from:
   - Speculative optimizations based on runtime profiling
   - Aggressive inlining of the entire predict loop
   - Register allocation tuned to the actual execution profile
   - Elimination of array bounds checks after proving loop invariants

3. **AVX2 lockstep overhead**: The SIMD approach processes 4 instances through the same tree
   in lockstep. When instances diverge (hit leaves at different depths), active lanes waste
   cycles on the active-mask check. For trees with variable depth paths, the lockstep
   overhead can be significant.

4. **Small dataset effect**: With 6950 instances × 29 attributes, the entire working set
   (~1.6 MB) fits in L2 cache. At this scale, memory access patterns matter less than
   instruction efficiency. The JIT's instruction-level optimizations dominate.

## What Was Implemented

- **Phase 1 (Scalar)**: `predict_scalar.c` -- cmov-friendly traversal loop, no bounds checks
- **Phase 2 (AVX2)**: `predict_avx2.c` -- processes 4 instances through same tree simultaneously
  using `_mm256_cmp_pd` + `_mm256_movemask_pd`, with active mask for completed lanes
- **Runtime dispatch**: CPUID-based AVX2 detection, function pointer dispatch via `g_batch_fn`
- **Zero-copy API**: `predictForBatchContiguous(MemorySegment, int)` + `flattenToOffHeap()`
  helper for pre-flattening data to off-heap memory
- **Correctness**: Bit-identical predictions verified against Java `ContiguousDfsForest` using
  `Double.doubleToRawLongBits()` comparison (single, batch, and zero-copy batch)

## Conclusion

Native C + AVX2 SIMD is **not a viable optimization** for random forest inference at this
scale. Java's JIT compiler produces equally good or better code for the simple tree traversal
loop. The 25-35% slowdown (depending on platform) is intrinsic to the native code, not the
FFM bridge overhead.

This result aligns with the broader pattern: for compute-bound loops with simple data access
patterns, the JVM's JIT is competitive with ahead-of-time compiled C. Native code wins when
it can exploit SIMD for truly data-parallel operations (e.g., matrix multiply, image
processing), but tree traversal is inherently serial per instance with divergent paths.

For the current workload, pure Java `ContiguousDfs` or `SeparateArraysBfs` remain the
fastest implementations.
