# Forest Variants — status & selection

The single source of truth for **which forest representation to use** and the lifecycle status of each.
Speed claims live in [PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md); prediction-correctness families
in [PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md). When you add a variant, add a row here.

**Family** — `faithful` reproduces the trained model's probabilities exactly (sum-then-normalize over
the real, generally non-unit leaf sums); `score` pre-normalizes each leaf then averages (faster, but
diverges from the trained model when leaves don't sum to 1). See PREDICTION-SEMANTICS.

**Status**
- **production** — safe to ship; measured competitive.
- **reference** — the canonical bit-exact baseline its family is validated against (also the test oracle).
- **experimental** — kept as a measurable baseline; not a default. May be situationally useful.
- **regression** — measured net-slower; kept only as a documented negative result. **Do not select.**

## Trainable

| Class | Status | Notes |
|---|---|---|
| `FasterForest` | reference | the trained model; `faithful` aggregation by construction. |
| `FasterForest2` | production | extended (interaction analysis); same aggregation. |

## Inference — faithful family (reproduces the trained model)

| `ForestType` | Status | Use when |
|---|---|---|
| `LegacyFlatBinaryForest` | reference / production | **default for faithful output**; what p2rank ships. Exact to the trained model (1e-15). |
| `ShortFlatBinaryForest` | production | faithful + ~50% memory (`float` arrays), near-identical speed. |
| `SuperShortLegacyFlatBinaryForest` | experimental | most compact (`short` indices); overflow risk on large forests. |

## Inference — score family (normalize-then-average; may diverge from trained model)

| `ForestType` | Status | Notes |
|---|---|---|
| `FlatBinaryForest` | reference / production | score-family reference; fastest layout on GraalVM (the deployment JIT). |
| `ContiguousDfsForest` | production | competitive with `Flat` on both JITs. |
| `SeparateArraysBfsForest` | production | competitive. |
| `FlatBinaryFloatForest` | experimental | `float` arrays; mid-pack, not bit-exact within its family. |
| `ContiguousBfsDoubleForest` | experimental | cache-layout study; mid-pack. |
| `InterleavedBfsForest` / `InterleavedBfsDoubleForest` | experimental | interleaved/BFS layout study; mid-pack on GraalVM (not the fastest, despite older README claims). |
| `BranchlessBfsForest` | regression | ~1.6× slower; always-both-sides work doesn't pay off. |
| `IlpDfsForest` / `IlpDfsFloatForest` | regression | bandwidth-bound; 1.2–2.4× slower, worse under load. |
| `BlockedIlpDfsForest` / `BlockedIlpDfsFloatForest` | regression | slowest; bandwidth-bound. |

## Inference — native (Panama FFM, score family, platform-dependent)

| `ForestType` | Status | Notes |
|---|---|---|
| `NativePanamaForest` / `NativePanamaFloatForest` | experimental | scalar native; mid-pack, does not beat the best Java here. Requires the native lib. |
| `NativePanamaForestAvx2` / `NativePanamaFloatForestAvx2` | regression | slower than their own scalar native — likely AVX-512/AVX2 downclock (PERFORMANCE-LESSONS lesson 5, unverified). |

## Selection guidance

- **Need exact agreement with the trained model / absolute probabilities** → `LegacyFlatBinaryForest`
  (or `ShortFlatBinaryForest` to save memory). This is the safe default and what p2rank uses.
- **Only ranking matters, want max throughput on GraalVM** → `FlatBinaryForest`.
- **Never select** a `regression` variant for production; they exist as measured baselines only.
