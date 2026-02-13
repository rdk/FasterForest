
<p align="center">
  <h1 align="center">🌲 FasterForest</h1>
  <p align="center">
    High-performance Random Forest library for Java with cache-optimized prediction
  </p>
</p>

<p align="center">
  <a href="/build.gradle"><img src="https://img.shields.io/badge/version-2.5.3-brightgreen.svg" alt="version 2.5.3"></a>
  <a href="https://github.com/rdk/FasterForest/actions/workflows/main.yml"><img src="https://github.com/rdk/FasterForest/actions/workflows/main.yml/badge.svg" alt="Build Status"></a>
  <a href="https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"><img src="https://img.shields.io/badge/License-GPL%20v2-blue.svg" alt="License: GPL v2"></a>
  <img src="https://img.shields.io/badge/Java-17+-orange.svg" alt="Java 17+">
</p>

---

**FasterForest** is a streamlined, high-performance **Random Forest library for Java**.
It provides multiple forest representations optimized for different speed/memory trade-offs, in particular **`InterleavedBfsForest`**
with cache-optimized layouts that achieve **3-4x speedup** over the standard flat array representation **`FlatBinaryForest`**.
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
| **`FlatBinaryForest`** | Parallel arrays | Baseline flat representation. |
| **`LegacyFlatBinaryForest`** | Parallel arrays + full class probs | Preserves both class probabilities for legacy compatibility. |
| **`ShortLegacyFlatBinaryForest`** | `float` arrays | ~50% memory reduction vs. double, minimal precision loss. |
| **`InterleavedBfsForest`** | Interleaved layout + BFS ordering | **Fastest.** 4 ints per node = 1 cache line per 4 nodes. BFS ordering keeps hot nodes in L1/L2. 3-4x faster than flat. |
| **`OptimizingFlatBinaryForest`** | Access-pattern reordering | Profiles node access counts, then reorders for cache locality. Multiple strategies (by tree, depth, count). |

### Conversion

All trainable forests implement `FlattableForest` and can be converted via `FasterForestConverter`:

```java
FasterForest ff = new FasterForest();
ff.buildClassifier(data);

BinaryForest fast = FasterForestConverter.convertFasterForest(ff, ForestType.InterleavedBfsForest);

double   score  = fast.predict(instance);
double[] scores = fast.predictForBatch(instances);
```

## 🚀 Performance

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

### InterleavedBfsForest optimizations

| Optimization | Speedup | Mechanism |
|---|---|---|
| **P1 - Interleaved layout** | 2-3x | Single `int[]` array, 4 ints/node (16 bytes). 4 nodes fit in one 64-byte cache line. |
| **P2 - BFS node ordering** | 1.2-1.4x | Breadth-first allocation. Top 6 levels (~63 nodes) fit in L1. |
| **P3 - Inlined traversal** | 1.1-1.3x | Local variable caching, `do-while` loop, multiply instead of divide. |

**Combined: ~3-4x** over the baseline `FlatBinaryForest`.

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

// Convert to InterleavedBfsForest (fastest, cache-optimized)
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
    ├── InterleavedBfsForest      # Cache-optimized (fastest)
    └── OptimizingFlatBinaryForest
```

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

GNU General Public License v2 — see [LICENSE.txt](LICENSE.txt).

