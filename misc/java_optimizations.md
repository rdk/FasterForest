# FlatBinaryForest Performance Optimization Notes

Typical workload: 200+ trees, depth ~12, batch size 3000, 50 attributes.

## Memory Budget (for typical workload)

| Data                           | Size    | Fits in  |
|--------------------------------|---------|----------|
| 1 instance (50 attrs)          | 400 B   | L1       |
| sums[3000]                     | 24 KB   | L1       |
| All instances 3000x50          | 1.2 MB  | L2/L3    |
| 1 tree (~2500 nodes) x 4 arrays| ~50 KB | L2       |
| All trees (200 x ~2500 nodes)  | ~10 MB  | L3 only  |

## Cache Behavior by Tree Depth

For 200 trees of depth 12 with batch size 3000:

| Level | Nodes per tree | Hits per node (of 3000) | Cache behavior         |
|-------|----------------|-------------------------|------------------------|
| 0-5   | ~63            | 3000 -> 47              | Hot -- pinned in L1    |
| 6-8   | ~448           | 47 -> 6                 | Warm -- L2 hits        |
| 9-12  | ~3500          | 6 -> 1                  | Cold -- L2/L3 misses   |

Deep nodes are the primary bottleneck.

## Optimizations

### P0: Bug fix (=  vs +=)

`FlatBinaryForest.predictForBatch` line 135 used `=` instead of `+=`, causing only the
last tree's prediction to survive. The single-instance `predict()` was correct.

### P1: Interleaved node data layout (biggest win, ~2-3x)

**Problem:** Current layout uses 4 parallel arrays (`childLeft[]`, `childRight[]`,
`attributeIndex[]`, `splitPoint[]`). Each node access during traversal fetches from 4
separate memory regions -- 4 cache line loads for 20 bytes of useful data (92% wasted
bandwidth).

**Solution:** Pack all node data into a single `int[]` array with stride 4:

```
nodeData[node*4 + 0] = childLeft
nodeData[node*4 + 1] = childRight
nodeData[node*4 + 2] = attributeIndex
nodeData[node*4 + 3] = Float.floatToRawIntBits(splitPoint)
```

One node = 16 bytes. Four nodes fit in a single 64-byte cache line. Each node access
requires 1 cache line load instead of 4.

Float precision for split points is sufficient for random forest split comparisons.
The decision boundary may shift by ~1e-7 relative to double, which is negligible
for ensemble predictions.

**Expected impact:** For cold nodes (levels 9-12), reduces cache misses by ~4x.
With 3000 instances x 200 trees x ~5 cold steps x 3 saved misses x ~10ns:
eliminates ~90ms of stall per batch.

### P2: BFS node layout in builder (~1.2-1.4x)

**Problem:** DFS pre-order allocation scatters sibling subtrees far apart. The right
child's subtree starts only after the entire left subtree is allocated.

**Solution:** Use BFS (breadth-first) ordering when allocating node indices. This
places nodes by depth level: root, then level-1 pair, then level-2 quad, etc.

**Benefit:** Top ~6 levels (63 nodes = 252 bytes in interleaved format) occupy ~4
consecutive cache lines, practically guaranteed to be pinned in L1 throughout all
3000 instances. Compounds with P1.

### P3: Inlined traversal + local variable caching (~1.1-1.3x)

**Problem:** `predictTree()` is a separate method called per tree-instance pair.
Array fields are accessed through `this` reference on every use.

**Solution:**
- Inline traversal directly into `predictForBatch` loop
- Cache array references as local variables (JIT hint: no aliasing)
- Use `do-while (node >= 0)` instead of `while(true)` + separate `if` check
- Multiply by `1/numTrees` instead of dividing (mul faster than div)

LegacyFlatBinaryForest already does the local-var caching in `predictTreeClassProbs`.
FlatBinaryForest does not.

### P4: Thread parallelism (Nx for N cores, orthogonal)

Partition instances across threads. Each thread processes all 200 trees for its
instance slice. Zero contention since each thread writes to a disjoint `sums[]` region.
With 3000 instances, ~375-750 instances per thread is enough to amortize overhead.

### P5: Loop tiling for large batches (~1.1-1.2x, marginal at batch=3000)

Tile the inner instance loop to keep `sums[tile]` and `instances[tile]` in L1.
At batch=3000, `sums[]` is only 24 KB (fits in L1 anyway), so tiling provides
minimal benefit. Would matter at batch sizes 10K+.

### P6: Float split points / scores (~1.1-1.2x)

Halves cache footprint for splitPoint and score arrays. Subsumes into P1 when using
the interleaved `int[]` layout with `Float.floatToRawIntBits`.

## Loop order: tree-major is correct

Current outer=trees, inner=instances is the right choice. Tree-major keeps one tree's
~50KB of node data warm in L2 across all 3000 instances. Instance-major would require
jumping between 200 different trees per instance, thrashing L2 with 10MB of tree data.

## Combined expected impact

| Optimization | Speedup   | Effort |
|-------------|-----------|--------|
| P1 interleave | 2-3x    | Medium |
| P2 BFS        | 1.2-1.4x | Medium |
| P3 inline     | 1.1-1.3x | Small  |
| P4 threads    | Nx        | Medium |
| **P1+P2+P3**  | **~3-4x** |       |

## Implementation

P1+P2+P3 are implemented in `InterleavedBfsForest.java`. Builder method is in
`FlatBinaryForestBuilder.buildInterleavedBfsForest()`.
