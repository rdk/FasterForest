# Resolved Bugs

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
