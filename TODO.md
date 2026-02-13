# TODO — Known Bugs

## Medium

### Missing `m_ZeroR` guard in `distributionForAttributes` (FasterForest and FasterForest2)

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

**Fix:** Add `m_ZeroR` null check to `distributionForAttributes()`, `predict()`, and
`predictForBatch()`, similar to how `distributionForInstance()` handles it. For the
prediction methods, the ZeroR model should return a constant probability for all instances.


### `getMaxDepth()` returns training depth limit, not actual tree depth (FasterForest)

**File:** `src/main/java/cz/siret/prank/fforest/FasterForest.java:321`

```java
public int getMaxDepth(){
    return m_MaxDepth;
}
```

`m_MaxDepth` is the user-configured training depth limit parameter (default `0`, meaning
unlimited). The `BinaryForest` interface contract expects `getMaxDepth()` to return the
actual maximum depth of the built trees. All other implementations (`FlatBinaryForest`,
`InterleavedBfsForest`, etc.) compute and return the real depth. FasterForest even has a
`calculateMaxTreeDepth()` method (line 800) that does the right thing, but `getMaxDepth()`
doesn't use it.

When no depth limit is set (the common case), `getMaxDepth()` returns `0`, which is
semantically opposite to the actual depth (~12+). Any caller using the `BinaryForest`
interface generically (e.g. to compare depths or allocate depth-sized buffers) will get
a wrong result.

**Fix:** `getMaxDepth()` should return `calculateMaxTreeDepth()` (possibly cached lazily).


## Low

### `getNumTrees()` returns configured count, not actual built count (FasterForest)

**File:** `src/main/java/cz/siret/prank/fforest/FasterForest.java:225`

```java
public int getNumTrees(){
    return m_numTrees;
}
```

Returns the parameter `m_numTrees` rather than the number of classifiers actually held in
`m_bagger`. Before `buildClassifier` is called, this returns the default (100) even though
no trees exist. If `setNumTrees()` is called after building, the returned value would
no longer match the actual forest. Other `BinaryForest` implementations return the
structural count.

**Fix:** After building, delegate to `m_bagger.getNumIterations()` or similar.
