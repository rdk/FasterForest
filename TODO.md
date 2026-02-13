# TODO — Known Bugs


## High

### #1 NPE in `buildRootTree` when `classIndex == 0` (FasterForest)

**File:** `src/main/java/cz/siret/prank/fforest/FasterTreeTrainable.java:1020`

```java
buildTree(data.sortedIndices, 0, data.sortedIndices[0].length-1,
    classProbs, attIndicesWindow, 0);
```

During training, `createInBagSortedIndices()` populates `sortedIndices[a]` for every
attribute `a` except the class attribute (`a == classIndex`), leaving
`sortedIndices[classIndex]` as `null`. The `buildRootTree` method uses
`data.sortedIndices[0].length - 1` as the `endAt` parameter. If the class attribute
happens to be at index 0, this dereferences `null` and throws a `NullPointerException`.

The typical case (class attribute is the last column) works fine because
`sortedIndices[0]` is a non-class attribute. But Weka datasets can have the class
attribute at any position, so this is a real crash scenario.

**Suggested fixes:**
- **(a) (Recommended)** Replace `data.sortedIndices[0].length - 1` with `data.numInBag - 1`.
  Direct, no iteration needed, and `numInBag` is always correctly set.
- (b) Find the first non-null `sortedIndices` entry and use its length.
  More defensive but adds unnecessary iteration.


### #2 `createInBagSortedIndicesNew` overflows `attInSortedIndices` (FasterForest2)

**File:** `src/main/java/cz/siret/prank/fforest2/DataCache2.java:275-304`

In `resample()`, the array `attInSortedIndices` is allocated with size
`nAttInSortedIndices`, which counts only non-nominal attributes among the selected
features. But `createInBagSortedIndicesNew()` iterates over ALL `selectedAttributes`
(including nominal ones) and writes every one of them into `attInSortedIndices`:

```java
for (int a : selectedAttributes) {
    attInSortedIndices[idx] = a;
    ++idx;
    ...
}
```

If any nominal attributes are among the selected features, `idx` exceeds the array
length, causing an `ArrayIndexOutOfBoundsException`. The special case
`allCategorical` allocates size 1, which also overflows with multiple nominal
attributes.

Currently harmless because the classifier only enables `NUMERIC_ATTRIBUTES` in its
capabilities, so nominal attributes never appear. But this is a latent crash if
nominal support is added.

**Suggested fixes:**
- **(a)** Skip nominal attributes in the `createInBagSortedIndicesNew` loop (matching
  the sizing logic in `resample()`).
- **(b) (Recommended)** Size `attInSortedIndices` to `selectedAttributes.length` so it
  can hold all selected attributes regardless of type. Simpler and more robust.


## Medium

### #3 Biased Fisher-Yates shuffle (FasterForest and FasterForest2)

**Files:**
- `src/main/java/cz/siret/prank/fforest/FastRfUtils.java:239-254`
- `src/main/java/cz/siret/prank/fforest2/FastRfUtils.java:243-258`

The Knuth (Fisher-Yates) shuffle implementation picks the swap target from the
entire array instead of the remaining unshuffled portion:

```java
for (int i = 0; i < numElems - 1; i++) {
    int next = rng.nextInt(numElems);          // BUG: biased
    // should be: rng.nextInt(numElems - i) + i
    int tmp = permutation[i];
    permutation[i] = permutation[next];
    permutation[next] = tmp;
}
```

A correct Fisher-Yates shuffle must pick `next` from `[i, numElems)`. Picking from
`[0, numElems)` produces a non-uniform distribution over permutations (the "naive
shuffle" bias, well-documented in the literature).

This is used for:
- Feature importance computation (scrambling attribute values to measure OOB error
  increase). The bias means importance values have a small systematic error.
- Feature subset selection in FasterForest2 (`DataCache2.resample()`). The bias
  means certain attribute subsets are slightly more/less likely than they should be.

**Fix:** Change to `int next = rng.nextInt(numElems - i) + i;` (in both files).


### #4 OOB error counts instances that are never out-of-bag (FasterForest and FasterForest2)

**Files:**
- `src/main/java/cz/siret/prank/fforest/FastRfBagging.java:299-314`
- `src/main/java/cz/siret/prank/fforest2/FastRfBagging.java:303-325`

When computing OOB error via `computeOOBError()`, every instance contributes to the
error calculation, even those that happen to be in-bag for ALL trees. For such
instances, the OOB vote array (`classProbs`) is all zeros, so
`Utils.maxIndex(classProbs)` returns 0 (the first class). If the true class is not 0,
the instance is counted as misclassified.

With 100 trees and standard 100% bootstrap, the probability of a single instance
being in-bag for all trees is approximately `(1 - 1/e)^100`, which is vanishingly
small. But with small forests (e.g. 10 trees) or reduced bag sizes, this becomes
more likely and inflates the OOB error.

**Suggested fixes:**
- **(a) (Recommended)** In `computeOOBError`, skip instances where total vote weight
  is zero (i.e. sum of `classProbs` == 0). Simple check, no changes to vote
  collectors needed.
- (b) In the OOB vote collectors (`VotesCollector` / `VotesCollectorDataCache`),
  return a sentinel value (e.g. -1) when `numVotes == 0`, and handle it in the
  caller. More invasive but makes the "no votes" case explicit.


### #5 `computeInteractions` overwrites importances via redundant recomputation (FasterForest2)

**File:** `src/main/java/cz/siret/prank/fforest2/FastRfBagging.java:504-527`

When both feature importance and interactions are enabled, the call sequence is:

1. `buildClassifier()` calls `computeImportances()` — computes and stores
   `m_FeatureImportances`, advancing the shared `random` RNG state.
2. `buildClassifier()` calls `computeInteractions()` — which internally calls
   `computeImportances()` AGAIN (line 511), overwriting `m_FeatureImportances`
   with different values (because `random` has advanced since step 1).

The interaction formula at line 522 subtracts `importance[i] + importance[j]` from
the joint scramble error, using the second-computation importances. But the
importances returned to the user via `getFeatureImportances()` are also these
second-computation values, not the original ones from step 1.

This means: (a) the reported importances are computed with a non-fresh RNG state,
and (b) the importances and interactions come from different scramble sequences.

**Suggested fixes:**
- **(a) (Recommended)** Guard the call in `computeInteractions()`: only call
  `computeImportances()` if `m_FeatureImportances == null`. This avoids the
  redundant recomputation and keeps importances consistent.
- (b) Save `m_FeatureImportances` before the call in `computeInteractions()` and
  restore it after. Preserves the original values but still wastes RNG state.
- (c) Give `computeInteractions()` its own `Random` instance (separate from the
  shared field). Fully isolates the two computations but adds complexity.


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


## Low

### #8 `giniConditionedOnRows` division by zero on empty branch (FasterForest2)

**Files:**
- `src/main/java/cz/siret/prank/fforest2/SplitCriteria.java:71-89`
- `src/main/java/cz/siret/prank/fforest2/SplitCriteria.java:107-117`

In `giniConditionedOnRows` and `giniConditionedOnRowsLR2`:

```java
returnValue += sumForBranch - auxSum / sumForBranch;
```

If a branch has zero total weight (`sumForBranch == 0`), this divides by zero,
producing NaN that propagates through all subsequent comparisons.

In practice, the split evaluation loop in `distributionSequentialAtt` always has at
least one instance per branch (it starts scanning from `startAt + 1`), so the zero
case should not arise during normal split search. However, the guard is missing and
could trigger in edge cases or if the function is called from a different context.

**Fix:** Add `if (sumForBranch == 0) continue;` before the division.


### #9 `giniOverColumns` division by zero on empty data (FasterForest2)

**File:** `src/main/java/cz/siret/prank/fforest2/SplitCriteria.java:169-183`

```java
return total - auxSum / total;
```

Same issue as above — divides by `total` which is zero when all weights are zero.
Only triggers with completely degenerate data (all instance weights zero).

**Fix:** Return 0 when `total == 0`.


## Resolved

### #10 `getMaxDepth()` returns training depth limit, not actual tree depth (FasterForest) — FIXED

`getMaxDepth()` now delegates to `calculateMaxTreeDepth()` after building. New
`getMaxDepthLimit()` method returns the training depth limit (`m_MaxDepth`). Internal
callers (options serialization, toString) updated to use `getMaxDepthLimit()`.

### #11 `getNumTrees()` returns configured count, not actual built count (FasterForest) — FIXED

`getNumTrees()` now delegates to `m_bagger.getClassifiers().length` after building.
Before building, falls back to the configured `m_numTrees` parameter.
