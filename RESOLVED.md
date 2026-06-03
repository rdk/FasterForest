# Resolved Bugs

### #7 Missing `m_ZeroR` guard in `distributionForAttributes` (FasterForest and FasterForest2) — FIXED

**Files:**
- `src/main/java/cz/siret/prank/fforest/FasterForest.java`, `.../fforest2/FasterForest2.java`
- `src/main/java/cz/siret/prank/fforest/FastRfBagging.java`, `.../fforest2/FastRfBagging.java`

When the training data has only the class attribute, `buildClassifier` falls back to a `m_ZeroR`
model and previously returned early without initializing `m_bagger`. `distributionForInstance()`
was guarded, but `distributionForAttributes()`, `predict()`, and `predictForBatch()` called the null
`m_bagger` and threw a `NullPointerException`.

Fixed by absorbing the degenerate case into the object graph instead of special-casing the prediction
paths: the ZeroR branch now installs a real `FastRfBagging` holding a single leaf tree that carries
ZeroR's constant class prior (`FastRfBagging.initConstantLeaf`, added in both packages). Sum-then-
normalize over one leaf reproduces the prior exactly, so the hot prediction methods stay byte-for-byte
unchanged — benchmarked indistinguishable from baseline on GraalVM (`singlePredict`/`Original`, 5
forks: baseline 2467 ± 25, fixed 2480 ± 22 ns/op, overlapping CIs). `distributionForInstance` keeps
its existing `m_ZeroR` guard.

Covered by `FasterForestZeroRTest` (FF1 + FF2). Limitation: fixes models built after this change; an
already-serialized degenerate model still deserializes with a null `m_bagger` (deserialization does
not re-run `buildClassifier`).

### #6 FF2 split evaluation skips potentially optimal split points (FasterForest2) — FIXED

**File:** `src/main/java/cz/siret/prank/fforest2/FasterForest2Tree.java:746`

Removed the `prevInstClass != data.instClassValues[inst]` condition from the split
evaluation loop in `distributionSequentialAtt`. The original code required both a
class-label change and an attribute-value change between consecutive sorted instances
before evaluating a candidate split. This was intended as an optimization (if classes
are the same at the boundary, the marginal Gini change at that exact point is zero),
but it is incorrect: Gini impurity depends on the cumulative class distribution in
each branch, not just the boundary instances. Moving a run of same-class instances
from one branch to the other can improve the overall split.

Example: instances sorted by attribute A with classes `[0, 0, 1, 0, 0]` and distinct
values. The old code only evaluated splits at positions 1|2 (0->1) and 2|3 (1->0),
missing the optimal split at position 3|4 (putting `[0,0,1,0]` vs `[0]`).

The fix aligns FF2 with the standard Random Forest algorithm and with FasterForest
(v1), which correctly evaluates all attribute-value boundaries. The cumulative
distribution (`currDistL0/1`, `currDistR0/1`) was already updated unconditionally
before the condition check, so no other changes were needed.

**Performance impact analysis** (estimated for 500 trees, 50 attributes, depth 15,
2M instances):
- Training time increases ~10-20% due to more Gini evaluations per attribute per node
- Each extra Gini evaluation is ~4ns of pure ALU work (2 multiplies, 2 adds, 1 divide,
  1 compare in `giniConditionedOnRowsLR2`)
- The cost is negligible relative to the O(n log n) sorting that dominates split finding
- No impact on inference speed (only training is affected)
- Potentially better model quality due to finding truly optimal splits

### #8 `giniConditionedOnRows` division by zero on empty branch (FasterForest2) — FIXED

Added zero guards to `giniConditionedOnRows`, `giniRow`, and `giniConditionedOnRowsLR2`
in `SplitCriteria.java`. Empty branches now contribute 0 instead of NaN.

### #9 `giniOverColumns` division by zero on empty data (FasterForest2) — FIXED

Added `if (total == 0) return 0;` guard in `giniOverColumns` in `SplitCriteria.java`.

### #3 Biased Fisher-Yates shuffle (FasterForest and FasterForest2) — FIXED

Changed `rng.nextInt(numElems)` to `rng.nextInt(numElems - i) + i` in both
`FastRfUtils.java` files. Now produces correct uniform permutations.

### #4 OOB error counts instances that are never out-of-bag (FasterForest and FasterForest2) — FIXED

VotesCollector and VotesCollectorDataCache now return `NaN` when `numVotes == 0`.
All `computeOOBError` loops skip `NaN` votes. Fixed in both FF and FF2 (8 files).

### #5 `computeInteractions` overwrites importances via redundant recomputation (FasterForest2) — FIXED

Guarded `computeImportances()` call in `computeInteractions()` with
`if (m_FeatureImportances == null)`. Importances are no longer recomputed when
both importances and interactions are enabled.

### Native `predictForBatch` IndexOutOfBoundsException — FIXED

`NativePanamaForest.predictForBatch()` threw `IndexOutOfBoundsException` when copying
instance data to off-heap memory. Weka's `numAttributes()` includes the class attribute,
so the forest stored N+1 as the row width. But instance arrays passed at prediction time
contain only the N feature values (class excluded). The code copied `numAttributes` doubles
per row from a source array that was one element shorter.

The pure-Java `ContiguousDfsForest` was unaffected because it accesses `inst[attr]` by index
without any stride.

**Fix:** Copy `instances[i].length` doubles per row instead of `numAttributes`. The off-heap
buffer is still allocated with `numAttributes` stride (required by the C batch functions),
and unused positions remain zero-filled. Tree attribute indices never reference the class
column, so padded positions are never read. Applied to `predictForBatch()` and
`flattenToOffHeap()`.

### #1 NPE in `buildRootTree` when `classIndex == 0` (FasterForest) — DOCUMENTED

Warning comment added in `FasterTreeTrainable.java:1021`. OK in practice because Weka
convention places the class attribute last. Would break if `classIndex == 0`.

### #2 `createInBagSortedIndicesNew` overflows `attInSortedIndices` (FasterForest2) — DOCUMENTED

Warning comment added in `DataCache2.java:278`. OK in practice because the classifier
only enables `NUMERIC_ATTRIBUTES`. Would break if nominal attribute support is added.

### #10 `getMaxDepth()` returns training depth limit, not actual tree depth (FasterForest) — FIXED

`getMaxDepth()` now delegates to `calculateMaxTreeDepth()` after building. New
`getMaxDepthLimit()` method returns the training depth limit (`m_MaxDepth`). Internal
callers (options serialization, toString) updated to use `getMaxDepthLimit()`.

### #11 `getNumTrees()` returns configured count, not actual built count (FasterForest) — FIXED

`getNumTrees()` now delegates to `m_bagger.getClassifiers().length` after building.
Before building, falls back to the configured `m_numTrees` parameter.

## Findings from BinaryForestInferenceTest

Added `BinaryForestInferenceTest` — 13 tests verifying prediction equivalence across
all 17 BinaryForest implementations. Three behavioral properties were confirmed:

### Float split-point casting causes path divergence (not a bug)

Float-precision forests (InterleavedBfsForest, FlatBinaryFloatForest, IlpDfsFloatForest,
NativePanamaFloatForest) cast double split points to float during construction. When an
instance's feature value falls between the double and float representations of a split
point, the tree traversal takes a different path, producing a completely different leaf
score. With 128 trees, a single divergent tree produces a prediction difference of
`1/128 = 0.0078125` — far beyond any accumulation-precision tolerance.

**Consequence:** Float forests cannot be validated against double-precision references.
They must be compared against each other using a float-family reference
(FlatBinaryFloatForest). All float forests agree with each other within `1e-6`.

### Legacy ground truth has ULP-level accumulation differences (not a bug)

`FasterForest.distributionForInstance()` and `LegacyFlatBinaryForest.distributionForInst()`
both implement the same legacy class-probs computation but through different code paths
(tree objects vs flattened arrays). Differences appear at the 16th significant digit
(e.g., `0.07059420220059569` vs `0.07059420220059565`). Tolerance of `1e-15` is required,
consistent with the existing `DELTA_15` in `FasterForestTest`.

### Native forests do not track `maxDepth` (not a bug)

`NativePanamaForest.getMaxDepth()` and `NativePanamaFloatForest.getMaxDepth()` return
`-1`. The native C implementation does not store tree depth metadata. Structure validation
for native forests checks only `numTrees` and `numAttributes`.

## Findings from TrainingDeterminismTest

### Training IS deterministic given fixed seed (confirmed, not a bug)

Thorough trace of the complete RNG chain for both FasterForest (FF1) and FasterForest2
(FF2) confirmed that training is fully deterministic given the same seed and input data.
The chain was verified through: seed propagation in `FastRfBagging`, parallel sort in
`IndexParallelSorter` (stable TimSort with stable merge), bootstrap sampling in
`DataCache.resample()`/`DataCache2.resample()`, `getRandomNumberGenerator()` data
signature mixing, and tree building in `FasterTreeTrainable`/`FasterForest2Tree`.

Key properties that ensure determinism:
- Seeds are pre-computed into an `int[]` array sequentially before parallel tree building
- Each tree gets its own `Random` instance from a deterministic seed
- The parallel sort (`IndexParallelSorter`) uses `IndexTimSort` (stable) with stable
  merge (`<=` takes left on ties); partition boundaries are determined by array sizes,
  not thread scheduling
- No shared mutable state exists between parallel tree builders; each tree's `DataCache`
  is a shallow copy with its own `inBag`, `instWeights`, and `whatGoesWhere` arrays
- The mother `DataCache`'s `vals` and `sortedIndices` are only read during tree building

`TrainingDeterminismTest` (7 tests) verifies: same seed → bit-identical tree structures
and predictions for both FF1 and FF2, including across different thread counts (1 vs 4).

### `getRandomNumberGenerator()` could hit null `sortedIndices[classIndex]` — HARDENED

`DataCache.getRandomNumberGenerator()` and `DataCache2.getRandomNumberGenerator()` pick
a random attribute index via `r.nextInt(numAttributes)` to compute a data signature from
`sortedIndices`. Since `sortedIndices[classIndex]` is null (the class attribute is skipped
during sorting), this could select a null array. `Arrays.hashCode(null)` returns 0, which
is deterministic but loses the data signature mixing (the RNG seed degenerates to just the
input seed).

**Fix:** Skip classIndex when picking the attribute: `if (attIdx == classIndex) attIdx =
(attIdx + 1) % numAttributes`. Applied to both `DataCache.java` and `DataCache2.java`.

### Stale comment in `DataCache2.getRandomNumberGenerator()` — FIXED

The comment `"ignore data signature since sortedIndices are not sorted in a stable way"`
was wrong — the sort IS stable (IndexTimSort). The comment was likely a historical artifact
from when quickSort was used. Updated to accurately document the current behavior.

### ForkJoinPool resource leak in DataCache constructors — FIXED

Both `DataCache` and `DataCache2` constructors created a `ForkJoinPool` for parallel
sorting but never shut it down. Added `pool.shutdown()` after the sorting loop.
