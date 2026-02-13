# TODO — Known Bugs


## Medium

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


See [RESOLVED.md](RESOLVED.md) for fixed and documented items.
