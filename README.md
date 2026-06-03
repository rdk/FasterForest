
<p align="center">
  <h1 align="center">🌲 FasterForest</h1>
  <p align="center">
    High-performance Random Forest library for Java with cache-optimized prediction
  </p>
</p>

<p align="center">
  <a href="/build.gradle"><img src="https://img.shields.io/badge/version-2.11.0-brightgreen.svg" alt="version 2.11.0"></a>
  <a href="https://github.com/rdk/FasterForest/actions/workflows/main.yml"><img src="https://github.com/rdk/FasterForest/actions/workflows/main.yml/badge.svg" alt="Build Status"></a>
  <a href="https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"><img src="https://img.shields.io/badge/License-GPL%20v2-blue.svg" alt="License: GPL v2"></a>
  <img src="https://img.shields.io/badge/Java-17+-orange.svg" alt="Java 17+">
</p>

---

**FasterForest** is a streamlined, high-performance **Random Forest library for Java**.
It provides multiple interchangeable forest representations optimized for different speed/memory
trade-offs. The fastest layout is **JIT-dependent** and benchmark-driven — see
[PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md) for current, JVM-stated rankings, and
[VARIANTS.md](VARIANTS.md) for which representation to pick.
For inference, **`FlatBinaryForest`** alone is already **1.5x** faster than Fran Supek's **`FastRandomForest`**,
which itself was a significant improvement over Weka's standard **`RandomForest`**.


> **Note:** Designed for numeric attributes only. Does not support nominal attributes or missing values.

## 🏛️ Random Forest Implementations

FasterForest provides a family of interchangeable `BinaryForest` implementations.
Train with one of the trainable classifiers, then convert to an optimized format for fast inference.

### Trainable Forests

| Implementation | Description |
|---|---|
| **`FasterForest`** | Main trainable classifier. Multi-threaded bagging, configurable tree depth, feature subsampling, OOB error, feature importance. |
| **`FasterForest2`** | Extended variant with dropout importance, pairwise feature interaction analysis. |

### Optimized Prediction Forests

These are inference-only representations converted from a trained forest.

| Implementation | Key Optimization | Notes |
|---|---|---|
| **`FlatBinaryForest`** | Parallel arrays | Baseline flat representation. Score-based (see note below). |
| **`LegacyFlatBinaryForest`** | Parallel arrays + full class probs | Keeps per-leaf class probabilities and reproduces the trained model exactly (1e-15). Score-preserving "legacy" family. |
| **`ShortLegacyFlatBinaryForest`** | `float` arrays | ~50% memory reduction vs. double, minimal precision loss. Legacy family. |
| **`InterleavedBfsForest`** | Interleaved layout + BFS ordering | 4 ints per node = 1 cache line per 4 nodes; BFS ordering keeps hot nodes in L1/L2. Score-based. (Ranking is JIT-dependent — see [PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md).) |
| **`OptimizingFlatBinaryForest`** | Access-pattern reordering | Profiles node access counts, then reorders for cache locality. Multiple strategies (by tree, depth, count). Constructed directly from a `LegacyFlatBinaryForest` (not exposed as a `ForestType`). |

> ⚠️ **Not all representations compute the same prediction.** They split into two families —
> *score-based* (`FlatBinaryForest` and all the Interleaved/Contiguous/Separate/Branchless/ILP/Float/
> Native variants) and *legacy class-probs* (`LegacyFlatBinaryForest`, `ShortLegacy…`, `SuperShortLegacy…`).
> Only the legacy family reproduces the trained `FasterForest`'s probabilities exactly; the score family
> can diverge because trained leaves generally do **not** sum to 1. If you depend on absolute
> probabilities (not just ranking), this matters — see **[PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md)**.

### Conversion

All trainable forests implement `FlattableForest` and can be converted via `FasterForestConverter`:

```java
FasterForest ff = new FasterForest();
ff.buildClassifier(data);

// FlatBinaryForest is the fastest layout on the deployment JIT (GraalVM); see VARIANTS.md / PERFORMANCE-LESSONS.md
BinaryForest fast = FasterForestConverter.convertFasterForest(ff, ForestType.FlatBinaryForest);

double   score  = fast.predict(instance);
double[] scores = fast.predictForBatch(instances);
```

## 🚀 Performance

> 📊 **The figures in this section are historical** — rough order-of-magnitude ratios vs.
> CDK/Weka/`FastRandomForest` from earlier runs. The single source of truth for current speed claims is
> **[PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md)** (JMH-measured, with the JVM stated), which also
> documents that the fastest layout is **JIT-dependent** (Graal favours `Flat`, HotSpot C2 favours the
> original tree), that the ILP/Blocked/Branchless variants are net regressions, and that native AVX2
> underperforms. Any perf number without a JVM + harness + date belongs there, not here.

### Inference speedup vs. predecessors

| Implementation | vs. Weka `RandomForest` | vs. `FastRandomForest` |
|---|---|---|
| **`FlatBinaryForest`** | ~15-30x | ~1.5x |
| **`InterleavedBfsForest`** | ~45-120x | ~5-6x |

Weka's `RandomForest` uses deep object trees with virtual dispatch per node.
Supek's `FastRandomForest` improved on that with streamlined data structures and multi-threaded training (10-20x over Weka).
FasterForest's flat representations eliminate pointer chasing entirely, and the interleaved BFS layout
further maximizes cache utilization.

### Training

| | vs. Weka `RandomForest` | vs. `FastRandomForest` |
|---|---|---|
| **Time** | ~13-27x | ~1.3x |
| **Memory** | ~4-10x | ~2x |

Supek's `FastRandomForest` introduced multi-threaded bagging and a shared data cache,
achieving 10-20x training speedup and ~2-5x memory reduction over Weka on multi-core machines.
`FasterForest` further streamlines the tree building code, reducing training time to ~75% and
memory usage to ~50% of `FastRandomForest`.

### Layout experiments

The repo carries many cache-layout variants (interleaved/BFS/DFS/ILP/native). Their *measured*
rankings — and the finding that the fastest one is JIT-dependent (and that several are net
regressions) — live in [PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md); their status (production /
experimental / regression) is in [VARIANTS.md](VARIANTS.md).

## 👨‍💻 Usage

### Training

```java
FasterForest forest = new FasterForest();
forest.setNumTrees(200);
forest.setNumFeatures(0);    // 0 = log2(M) + 1
forest.setMaxDepth(0);       // 0 = unlimited
forest.setNumThreads(0);     // 0 = auto-detect cores
forest.buildClassifier(data);
```

### Prediction

```java
// Single instance (returns P(class=1))
double score = forest.predict(featureVector);

// Batch prediction
double[] scores = forest.predictForBatch(featureMatrix);

// Full distribution via Weka API
double[] dist = forest.distributionForInstance(instance);
```

### Converting to optimized inference formats

```java
import cz.siret.prank.fforest.api.FasterForestConverter;
import cz.siret.prank.fforest.api.FasterForestConverter.ForestType;
import cz.siret.prank.fforest.api.BinaryForest;

// Convert to FlatBinaryForest (baseline flat representation)
BinaryForest flat = FasterForestConverter.convertFasterForest(forest, ForestType.FlatBinaryForest);

// Convert to InterleavedBfsForest (cache-optimized layout; ranking is JIT-dependent — see PERFORMANCE-LESSONS.md)
BinaryForest fast = FasterForestConverter.convertFasterForest(forest, ForestType.InterleavedBfsForest);

// All BinaryForest implementations share the same prediction API
double   score  = fast.predict(featureVector);
double[] scores = fast.predictForBatch(featureMatrix);
```

### Feature Importance

```java
forest.setComputeImportances(true);
forest.buildClassifier(data);
double[] importances = forest.getFeatureImportances();
```

### Build

```bash
./gradlew build
```

### Benchmarking

There are two harnesses (not interchangeable — see [DEVELOPMENT.md](DEVELOPMENT.md)):

- **JMH (use this for rankings)** — forks a fresh JVM per benchmark: `./jmh.sh` (or `./gradlew jmh`).
  Benchmark on **GraalVM** (the deployment JIT; rankings invert vs HotSpot C2).
- **Legacy** quick relative checks: `./benchmark.sh`.

> Prediction correctness is anchored by `PredictionGoldenTest` (a pinned baseline in the standard
> `./gradlew test` run), so hot-path refactors that change output are caught automatically.

Legacy harness:

```bash
./benchmark.sh                            # run with defaults
./benchmark.sh -r 20 -i 500              # 20 measured rounds, 500 iters each
./benchmark.sh -t 200 -d 15              # 200 trees, max depth 15
./benchmark.sh -r 20 -i 500 -t 200 -d 15 # all params
```

| Flag | Parameter | Default | Description |
|---|---|---|---|
| `-r` | `MEASURE_ROUNDS` | 10 | Number of measured rounds |
| `-i` | `ITERS_PER_ROUND` | 400 | Prediction iterations per round |
| `-t` | `NUM_TREES` | 100 | Number of trees in the forest |
| `-d` | `TREE_DEPTH` | 0 | Max tree depth (0 = unlimited) |

The benchmark Gradle task applies recommended JVM settings for stable results (`-server`, `-XX:-TieredCompilation`, `-XX:+AlwaysPreTouch`, `-XX:+UseParallelGC`).

You can also run benchmarks directly via Gradle:

```bash
./gradlew benchmark
./gradlew benchmark -Dbench.measureRounds=20 -Dbench.itersPerRound=500
```

## 🏗️ Architecture

```
cz.siret.prank.fforest
├── FasterForest              # Trainable classifier (v1)
├── FasterForest2             # Extended with interaction analysis (v2)
├── FasterTree                # Tree node (linked structure)
├── FasterTreeTrainable       # Tree building algorithm
├── FastRfBagging             # Bootstrap ensemble builder
└── api/
    ├── BinaryForest              # Core prediction interface
    ├── TrainableFasterForest     # Training interface
    ├── FlattableForest           # Conversion interface
    ├── FasterForestConverter     # Unified conversion factory
    ├── FlatBinaryForest          # Flat array forest
    ├── LegacyFlatBinaryForest    # Flat with full class probs
    ├── ShortLegacyFlatBinaryForest
    ├── InterleavedBfsForest      # Cache-optimized layout
    └── OptimizingFlatBinaryForest
```

## 📚 Documentation

| Doc | Owns (single source of truth for…) |
|---|---|
| `README.md` | what the library is and how to use it |
| [`VARIANTS.md`](VARIANTS.md) | every forest representation: family, status, when to use which |
| [`PREDICTION-SEMANTICS.md`](PREDICTION-SEMANTICS.md) | correctness — the score vs. legacy (faithful) prediction families |
| [`PERFORMANCE-LESSONS.md`](PERFORMANCE-LESSONS.md) | all speed claims + benchmarking methodology (JVM-stated) |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | architecture, how to benchmark, how to add a variant |
| `CLAUDE.md` | conventions / definition-of-done for agents and contributors |

## 🙏 Acknowledgments

FasterForest builds on the work of:

- **[FastRandomForest](http://code.google.com/p/fast-random-forest/)** by Fran Supek
  - the original re-implementation of Random Forest for Weka
  that introduced multi-threaded training and significant speed/memory improvements
  over the standard Weka RF.

- **[FastRandomForest 2.0](https://github.com/GenomeDataScience/FastRandomForest)** by
  the GenomeDataScience group 
  - extended version with feature interaction analysis
  and dropout-based importance, which forms the basis of `FasterForest2`.

- **[Weka](https://www.cs.waikato.ac.nz/ml/weka/)** machine learning framework by
  the University of Waikato.

## 📜 License

GNU General Public License v2 - see [LICENSE.txt](LICENSE.txt).

