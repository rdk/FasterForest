# TODO — Known Bugs


## Medium

### #6 FF2 split evaluation skips potentially optimal split points (FasterForest2)

**File:** `src/main/java/cz/siret/prank/fforest2/FasterForest2Tree.java:746`

```java
if (prevInstClass != data.instClassValues[inst]
    && dataValsAtt[inst] > dataValsAtt[prevInst]) {
```

The standard Random Forest algorithm evaluates a candidate split at every boundary
where consecutive instances (sorted by the split attribute) have distinct values.
FasterForest (v1) follows this correctly:

```java
if (data.vals[att][inst] > data.vals[att][prevInst]) {
```

FasterForest2 adds an extra condition: it also requires that the class label changes
between consecutive instances. The intent is an optimization — if classes are the
same at the boundary, the marginal Gini change at that exact point is zero.

However, this ignores cumulative effects. The Gini impurity depends on the entire
class distribution in each branch, not just the boundary instances. Moving a run of
same-class instances from right to left can improve the split even though no single
boundary within that run changes the class.

Example: instances sorted by attribute A with classes `[0, 0, 1, 0, 0]` and distinct
values. The code only evaluates splits at positions 1|2 (0->1) and 2|3 (1->0), but
the optimal split could be at position 3|4 (putting `[0,0,1,0]` vs `[0]`).

**Suggested fixes:**
- **(a) (Recommended)** Remove the `prevInstClass != data.instClassValues[inst]`
  condition to match the standard algorithm. The overhead of evaluating extra split
  points is small relative to the sorting cost.
- (b) Keep the optimization but update the cumulative Gini tracking even for skipped
  boundaries, so the next evaluated split uses the correct cumulative distribution.
  More complex, preserves the speedup for long same-class runs.


### #7 Missing `m_ZeroR` guard in `distributionForAttributes` (FasterForest and FasterForest2) — WONTFIX

**Files:**
- `src/main/java/cz/siret/prank/fforest/FasterForest.java:719`
- `src/main/java/cz/siret/prank/fforest2/FasterForest2.java:772`

When the training data has only a class attribute (no features), `buildClassifier` creates a
`m_ZeroR` fallback model and returns early without initializing `m_bagger`. The
`distributionForInstance()` method correctly checks `if (m_ZeroR != null)` and delegates to
the fallback. However, `distributionForAttributes()` does not check — it directly calls
`m_bagger.distributionForAttributes(...)`, which throws a `NullPointerException` because
`m_bagger` is null.

This also affects all code paths that go through `distributionForAttributes`:
- `predict()` in both FasterForest (line 829) and FasterForest2 (line 964)
- `predictForBatch()` in both, which calls `getTrees()` → `m_bagger.getClassifiersAsTrees()`
- `toFlatBinaryForest()` in both, which also accesses `m_bagger`

Not fixing — the `m_ZeroR` fallback and Weka dependency are planned for removal.


## Resolved

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
