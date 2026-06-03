# Development Guide

High-performance Random Forest library for Java. Trains a single model (FasterForest),
then converts it to one of 17 optimized inference-only representations via a unified
`BinaryForest` interface. Numeric attributes only.

## Project Structure

```
src/main/java/cz/siret/prank/
├── fforest/                    # Core training pipeline (v1)
│   ├── FasterForest.java       #   Main trainable classifier
│   ├── FasterTree.java         #   Tree node (linked, immutable after training)
│   ├── FasterTreeTrainable.java#   Tree building algorithm
│   ├── FastRfBagging.java      #   Bootstrap ensemble builder (multi-threaded)
│   ├── DataCache.java          #   Columnar data storage for training
│   ├── SplitCriteria.java      #   Gini-based split selection
│   └── api/                    #   Public API and inference forests
│       ├── BinaryForest.java   #     predict(double[]) / predictForBatch(double[][])
│       ├── TrainableFasterForest.java  # Training interface
│       ├── FasterForestConverter.java  # Factory: ForestType enum → BinaryForest
│       ├── FlatBinaryForest.java       # Score-based reference implementation
│       ├── LegacyFlatBinaryForest.java # Legacy class-probs reference
│       ├── InterleavedBfsForest.java   # Fastest Java: cache-optimized BFS, float
│       ├── ContiguousDfsForest.java    # DFS layout (base for native forests)
│       ├── IlpDfsForest.java           # ILP 4-lane batch prediction
│       ├── NativeLoader.java           # Dynamic .so/.dll loading
│       └── ... (17 forest types total)
│
├── fforest2/                   # Extended classifier (v2)
│   └── FasterForest2.java      #   Adds dropout importance, interaction analysis
│
└── ffutils/                    # Utilities (timing, sorting, normalization)

src/main/java22/cz/siret/prank/fforest/api/   # Panama FFM (Java 22+)
├── NativePanamaForest.java          # Scalar C prediction
├── NativePanamaForestAvx2.java      # AVX2 SIMD prediction
├── NativePanamaFloatForest.java     # Float scalar
└── NativePanamaFloatForestAvx2.java # Float AVX2

native/                         # C source for prediction kernels
├── CMakeLists.txt
├── include/fasterforest.h
└── src/predict_{scalar,avx2}[_float].c

src/test/java/cz/siret/prank/fforest/
├── FasterForestTest.java            # Training, conversion, round-trip tests
├── BinaryForestInferenceTest.java   # Prediction equivalence across all 17 types
└── PredictionSpeedBenchmark.java    # Throughput benchmark (all types)
```

## Key Abstractions

**TrainableFasterForest** — Training interface. Implementations: `FasterForest`, `FasterForest2`.
Exposes `getTrees()` returning `List<FasterTree>` (linked tree nodes with classProbs and scores).

**BinaryForest** — Inference interface. All 17 forest types implement this.
Core methods: `predict(double[])`, `predictForBatch(double[][])`, `getNumTrees()`, `getNumAttributes()`.

**FasterForestConverter** — One-line conversion from any `TrainableFasterForest` to any `ForestType`:
```java
BinaryForest forest = FasterForestConverter.convertFasterForest(trainedModel, ForestType.InterleavedBfsForest);
```

## Two Prediction Families

The forests split into two mathematically incompatible prediction families:

| Family | Normalization | Reference implementation |
|--------|---------------|--------------------------|
| **Score-based** | Each leaf pre-normalizes `p1/(p0+p1)`, then averages across trees | `FlatBinaryForest` |
| **Legacy class-probs** | Sums raw `classProbs[0]` and `classProbs[1]` across trees, then normalizes | `LegacyFlatBinaryForest` |

Score-based: FlatBinaryForest, all Interleaved/Contiguous/Separate/Branchless/ILP/Float/Native variants.
Legacy: LegacyFlatBinaryForest, ShortLegacyFlatBinaryForest, SuperShortLegacyFlatBinaryForest.

The two families agree only when leaves sum to 1, which is **not** the usual case — trained leaves
encode mean bootstrap multiplicity (mean ≈ 1.25–1.50 on the test dataset). Only the legacy family
reproduces the trained model exactly. For the full explanation (leaf normalization, what a non-unit
leaf sum means, and how this relates to FastRandomForest/Weka) see
[PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md).

Float-precision forests (InterleavedBfsForest, FlatBinaryFloatForest, IlpDfsFloatForest,
NativePanamaFloatForest) cast split points to float, which causes path divergence — some
instances traverse different tree branches. They must be compared against a float-family
reference, not the double reference. See RESOLVED.md for details.

## All 17 Forest Types

| ForestType enum | Precision | Layout | Notes |
|-----------------|-----------|--------|-------|
| FlatBinaryForest | double | DFS | Score-based reference |
| LegacyFlatBinaryForest | double | DFS | Legacy class-probs reference |
| ShortFlatBinaryForest | float | DFS | Legacy, ~50% memory |
| SuperShortLegacyFlatBinaryForest | float | DFS | Legacy, most compact |
| InterleavedBfsForest | float | BFS | Fastest Java (cache-optimized, 4 ints/node) |
| InterleavedBfsDoubleForest | double | BFS | Double-precision interleaved |
| ContiguousBfsDoubleForest | double | BFS | Contiguous per-tree |
| SeparateArraysBfsForest | double | BFS | Separate int[]/double[] arrays |
| BranchlessBfsForest | double | BFS | Branchless cmov traversal |
| ContiguousDfsForest | double | DFS | Base for native forests |
| IlpDfsForest | double | DFS | ILP 4-lane batch prediction |
| IlpDfsFloatForest | float | DFS | ILP 4-lane, float |
| FlatBinaryFloatForest | float | DFS | Float-precision flat |
| NativePanamaForest | double | DFS | C via Panama FFM |
| NativePanamaForestAvx2 | double | DFS | C + AVX2 SIMD |
| NativePanamaFloatForest | float | DFS | C float |
| NativePanamaFloatForestAvx2 | float | DFS | C float + AVX2 |

## Multi-Release JAR

The project targets Java 17 but uses Java 22 Panama FFM for native interop:

- **Main sources** (`src/main/java`): Compiled at Java 17. Native classes here are stubs
  that return `isAvailable() = false`.
- **Java 22 sources** (`src/main/java22`): Real Panama FFM implementations. Packaged in
  `META-INF/versions/22/` in the multi-release JAR.
- **Tests**: Compiled at Java 22 with java22 output on classpath before main output,
  so real native classes take precedence over stubs.

## Building and Testing

```bash
./gradlew clean assemble          # Build JAR (Java only)
./test.sh                         # Run all tests with summary
./test.sh --tests "*.BinaryForestInferenceTest"  # Specific test class
./benchmark.sh                    # Run speed benchmarks
./benchmark.sh -t 200 -r 10      # Custom: 200 trees, 10 rounds
```

See [BUILDING.md](BUILDING.md) for native library compilation.

## Benchmarking

Two harnesses, **not interchangeable**:

- **JMH** (`./jmh.sh` or `./gradlew jmh`, sources in `src/jmh`) — forks a fresh JVM per benchmark;
  use this for **rankings**. `ForestPredictBenchmark` reports `batchPredict` (µs/op) and `singlePredict`
  (ns/op) with `forestType` / `numTrees` / `batchSize` params; `NativeZeroCopyBenchmark` covers the
  off-heap native path.
- **Legacy** (`./benchmark.sh` or `./gradlew benchmark`, `PredictionSpeedBenchmark`) — one long-lived
  JVM, median of rounds; fast relative sanity checks only (it warm-profiles later forests, over-rating
  them).

Rules (also in [CLAUDE.md](CLAUDE.md)):

- Benchmark on **GraalVM** — it is the deployment JIT and rankings invert vs HotSpot C2.
- Compare **ratios on the same JVM**, never absolute ms across runs; don't mix the two harnesses in one
  table.
- A perf claim states JVM + harness + date and lives in
  [PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md), which is the single source of truth for speed.
- To benchmark on a specific JVM while the build pins a toolchain, point JMH at it:
  `./gradlew jmh -PjmhArgs="ForestPredict -jvm /path/to/java -jvmArgs=--enable-native-access=ALL-UNNAMED"`.

## Dependencies

- **weka-dev 3.9.6** — Weka ML framework (Instance, Instances, AbstractClassifier)
- **commons-lang3** — String/array utilities
- **hppc** — High-performance primitive collections
- **JUnit 4** — Tests

## Test Dataset

All tests use `src/test/resources/data/p2rank-train.arff.gz` — a binary classification
dataset from the P2Rank protein binding site prediction tool.

## Known Issues

See [TODO.md](TODO.md) for open bugs and [RESOLVED.md](RESOLVED.md) for fixed items.
