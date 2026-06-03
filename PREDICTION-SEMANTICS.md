# Prediction Semantics: Score-based vs. Legacy Class-Probabilities

This note documents a subtle but important point: the optimized inference forests do **not**
all compute the same prediction. They split into two mathematically distinct families that can
disagree on real models. Which one you pick is a correctness decision, not just a speed/memory one.

## TL;DR

- A trained leaf does **not** generally store a probability distribution that sums to 1. It stores
  the per-class in-bag weight **divided by the number of distinct in-bag instances in the leaf** —
  i.e. the *mean bootstrap multiplicity* of the leaf's instances. On a real model most leaves sum to
  **more than 1** (mean ≈ 1.25 at `bagSizePercent=55`, ≈ 1.50 at 100%).
- This is **deliberate**, not a calibration bug. The non-unit sum is the mechanism by which a tree's
  vote is weighted by how much bootstrap support backs the leaf the instance fell into. Weka's
  `RandomForest` normalizes every leaf to sum 1, which *removes* this weighting.
- **Legacy** forests (`LegacyFlatBinaryForest`, `ShortLegacyFlatBinaryForest`,
  `SuperShortLegacyFlatBinaryForest`) preserve this weighting and reproduce the trained
  `FasterForest`'s predictions to ULP precision (1e-15).
- **Score-based** forests (`FlatBinaryForest` and **all** the Interleaved / Contiguous / Separate /
  Branchless / ILP / Float / Native variants) collapse each leaf to `p1/(p0+p1)` up front, which
  discards the weighting and gives every tree an equal vote. They **diverge** from the trained model
  whenever leaves don't sum to 1 — which is the common case.

## What a leaf stores after training

During tree construction, a leaf's class array starts as the summed **in-bag weight** per class of
the instances reaching it, then is divided by the instance **count**
(`FasterTreeTrainable.java`, leaf-making at lines ~159 and ~286):

```java
// normalize by dividing with the number of instances (as of ver. 0.97)
if (sortedIndicesLength != 0) {
    for (int c = 0; c < classProbs.length; c++)
        classProbs[c] /= sortedIndicesLength;   // divide by COUNT, not by weight-sum
}
```

so

```
leaf[c]  = (summed in-bag weight of class c) / (# distinct in-bag instances)
leaf sum = (total in-bag weight in leaf)     / (# distinct in-bag instances)
         = mean in-bag weight  =  mean bootstrap multiplicity of the leaf's instances
```

Bootstrap resampling assigns weights by draw count (`DataCache.resample`: *"when an instance is
sampled multiple times, its weight increases to a multiple of the original weight"*). Because the
divisor is the **count** of distinct instances (not the weight sum):

- A leaf whose instances were each drawn **once** → sum **= 1**.
- A leaf whose instances were drawn **repeatedly** (high bootstrap support) → sum **> 1**.
- Leaf sum does **not** encode "more instances" — the count is divided out. It encodes the *average
  resampling weight* of whatever instances landed there.

The design intent is documented in `FasterTree.java` (~line 192):

> *"In Weka's RandomTree, the distributions were normalized so that all probabilities sum to 1; this
> would abolish the effect of instance weights on voting. In FasterForest … the distributions are
> normalized by dividing with the number of instances going into a leaf."*

and `FastRfBagging.java` (~line 60):

> *"some trees will have a heavier weight in the overall vote depending on the averaged weights of
> instances that ended in the specific leaf."*

`FasterForest` has an opt-in flag `m_ensureLeavesNormalized` (default **false**) that forces leaves
back to sum 1 — recovering Weka semantics. Its source comment calls the non-normalized behaviour the
*"lagacy bug"*, but it is really a modeling choice.

## The two aggregation families

For a binary problem, predicting the positive-class probability of one instance:

| Family | Per tree | Across trees | Reference impl. |
|---|---|---|---|
| **Score-based** | `score = p1 / (p0 + p1)` (leaf normalized to magnitude 1) | average the scores | `FlatBinaryForest` |
| **Legacy class-probs** | keep raw leaf `[p0, p1]` | sum `p0`, sum `p1`, then normalize once | `LegacyFlatBinaryForest` |

The trained model's own aggregation (`FastRfBagging.distributionForAttributes`) is **sum-then-
normalize**: it sums per-tree distributions and calls `Utils.normalize` once at the end. That is the
legacy family. The score family is **normalize-then-average**.

These two are **identical iff every leaf sums to 1**. When leaves carry non-unit bootstrap weight
(the common case), they differ: sum-then-normalize lets high-support leaves vote harder;
normalize-then-average gives every tree an equal vote.

### Worked example

Two trees vote on one instance. Tree A lands in a leaf `[0, 3]` (one positive instance drawn 3× in
the bootstrap → sum 3); tree B lands in a leaf `[1, 1]` (two instances, one per class → sum 2).

- **Legacy (sum-then-normalize):** sum across trees = `[1, 4]` → positive prob = `4/5 = 0.80`.
- **Score (normalize-then-average):** A → `3/3 = 1.0`, B → `1/2 = 0.5`; mean = `(1.0 + 0.5)/2 = 0.75`.

`0.80 ≠ 0.75`: the high-support leaf A (sum 3) outvotes B under legacy, but counts equally under
score. They coincide only when both leaves already sum to 1.

## Empirical: leaf sums on a real model

Measured on `src/test/resources/data/p2rank-train.arff.gz` (50 trees, seed 42):

| `bagSizePercent` | leaves | mean leaf sum | sum to exactly 1.0 | max |
|---|---|---|---|---|
| 55 (the benchmark config) | 8,088 | 1.25 | 41% | 4.0 |
| 100 (default bootstrap) | 11,018 | 1.50 | 28% | 6.0 |

So 60–72% of leaves do not sum to 1. The means match the theoretical mean bootstrap multiplicity
(draws ÷ distinct-in-bag ≈ `0.55/0.42 ≈ 1.3` and `1.0/0.63 ≈ 1.58`). The divergence between the two
families is therefore not a rounding artifact — it affects the majority of leaves.

## Which representation should I use?

- Need to **match the trained `FasterForest` exactly** (e.g. you calibrated thresholds against it, or
  consume absolute probabilities): use a **legacy** forest — `LegacyFlatBinaryForest` (double),
  `ShortLegacyFlatBinaryForest` / `SuperShortLegacyFlatBinaryForest` (compact, ~1e-6).
  `FasterForest.toFlatBinaryForest()` (no-arg) returns a legacy forest by default.
- Only the **ranking / relative order** matters, and you want maximum speed/compactness: a
  **score-based** forest is fine, but be aware its probabilities differ from the trained model.
- Importing a **Weka** `RandomForest`: use `WekaRandomForestConverter`. It pre-normalizes each leaf to
  sum 1 (so that sum-then-normalize reproduces Weka's per-tree-equal voting); after that the two
  families coincide.

## Relationship to FastRandomForest and Weka RandomForest

`FasterForest` is a streamlined descendant of Fran Supek's FastRandomForest, which itself reworked
Weka's `RandomForest`.

- **Weka `RandomForest`** normalizes each leaf/tree to sum 1 → every tree votes equally
  (normalize-then-average). This is the score family's semantics.
- **`FasterForest`** (and FastRandomForest from v0.97) divides leaves by instance count instead, so
  bootstrap support modulates per-tree vote weight, aggregated sum-then-normalize. This is the legacy
  family's semantics, and it is what the trainer actually produces.

In other words, the "legacy" name is about preserving *this library's own* trained-model behaviour;
the score family is closer to textbook Weka.

## Code references

- Leaf normalization during training: `FasterTreeTrainable.java` (~159, ~286), `FasterTree.java` (~192).
- Bootstrap weighting: `DataCache.java` `resample()` (~209–237); `FastRfBagging.java` (~60).
- Trained-model aggregation: `FastRfBagging.distributionForAttributes()` (~684).
- Score collapse: `FlatBinaryForestBuilder.getScoreFromProbs()` (~186); builder legacy switch (~118).
- Legacy aggregation: `LegacyFlatBinaryForest.predictForBatch()` / `predictClassProbs()`.
- Weka import normalization: `WekaRandomForestConverter.java` (~129).
- Equivalence tests & tolerances: `BinaryForestInferenceTest`, `FasterForestTest`,
  `WekaRandomForestConverterTest` (see also `RESOLVED.md`).
