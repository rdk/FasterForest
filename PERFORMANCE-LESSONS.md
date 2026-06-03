# Performance Lessons (Inference)

A field guide from optimizing and benchmarking forest **prediction** speed. It records what we
measured, what surprised us, and what is explicitly not worth doing, so future work does not relearn
it the hard way. Companion to [PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md) (which covers the
*correctness* families — legacy vs score — not speed).

The kernel: each tree is a flat array-indexed descent (`attributeIndex[node]`, compare to
`splitPoint[node]`, branch to `childLeft/childRight[node]`, repeat until a negative index hits a leaf).
Prediction is `O(trees × depth)` per instance, single-threaded and read-only on the forest arrays.

> [!IMPORTANT]
> **The deployment JVM is GraalVM.** Several conclusions below invert between Graal and HotSpot C2, so
> rankings are only meaningful on a stated JVM. Where it matters the JVM is named. Bit-exact behaviour
> is verified separately by the equivalence tests (`BinaryForestInferenceTest` etc.); this doc is about
> speed, and speed is JIT-dependent.

## Benchmark setup

- **Machine:** 32 logical CPUs (this box), AVX2-capable.
- **JVMs:** GraalVM 25.0.2 (Graal JIT — the deployment JVM) and Oracle JDK 25.0.2 (HotSpot C2). Same
  `25.0.2` build; only the JIT differs, which isolates the compiler as the variable.
- **Data:** `src/test/resources/data/p2rank-train.arff.gz` — 6,950 instances, 29 numeric features,
  binary class. Forest: 200 trees, max depth 26, seed 42, 5 features/split, bag 55%.
- **Harnesses:** JMH (`src/jmh`, forks a fresh JVM per benchmark) and the legacy
  `PredictionSpeedBenchmark` (one long-lived JVM, median of rounds). Every number below is labelled
  with its harness; the two are **not** interchangeable (lesson 7).
- **Caveat:** JMH figures are single-fork unless noted. Hardware perf counters were unavailable
  (`perf_event_paranoid=4`, no hsdis), so codegen attribution here is **behavioural** (controlled
  experiments), not from disassembly.

## Headline numbers

**Batch prediction, JMH, 8 threads, 69,500 rows (tiled), 200 trees, avg-time µs/op (lower is faster), GraalVM:**

| Forest | µs/op | | Forest | µs/op |
|---|---|---|---|---|
| **Flat** | **165,940** | | InterleavedBfs | 194,547 |
| ContiguousDfs | 169,991 | | BranchlessBfs | 318,083 |
| SeparateArraysBfs | 170,419 | | IlpDfs | 343,253 |
| ShortLegacy | 170,518 | | IlpDfsFloat | 363,198 |
| LegacyFlat | 180,610 | | BlockedIlpDfs | 385,110 |
| FlatFloat | 188,346 | | BlockedIlpDfsFloat | 401,094 |
| ContiguousBfsDouble | 188,989 | | | |
| InterleavedBfsDouble | 193,185 | | | |
| Original (FF) | 194,172 | | | |

**Same config, Graal vs C2 (µs/op) — the ranking inverts:**

| Forest | Graal | C2 |
|---|---|---|
| Flat | **165,940** | 201,845 |
| Original (FF) | 194,172 | **168,541** |

Single-prediction latency (8 threads, ns/op): `Flat` ≈ 1,966 (Graal) / 1,957 (C2); `BranchlessBfs` ≈
4,400 — compiler-stable, see lesson 6.

## Hard-won lessons

### The JIT

1. **The JIT picks the winner — benchmark on the deployment JVM.** On Graal, `Flat` is fastest and the
   original object-tree `Original` is mid-pack; on C2 it **flips** — `Original` is fastest and `Flat`
   drops ~22%. Graal optimises the flat *array-indexed* traversal better; C2 optimises the object
   *pointer-chase* better. A ranking measured on Temurin/C2 does not transfer to a GraalVM deployment.

2. **The Graal-vs-C2 gap on `Flat` is scalar codegen, not branches and not SIMD.** A controlled
   experiment (`BranchSensitivityBenchmark`: predict a batch of *identical* rows so every tree takes one
   fixed, perfectly-predicted path with an L1-resident working set, vs the real varied rows) showed the
   varied/uniform speedup ratio is **identical across JITs** (Graal 2.03×, C2 2.04×) — so neither JIT
   loses more to branch misprediction; both compile the descent branch the same way. Yet Graal stays
   ~13% faster on `Flat` even in the *uniform* regime, where branches and memory are removed as factors,
   leaving only instruction throughput. So it is per-node scalar code quality (bounds-check elimination /
   addressing on the indexed loads), **not** vectorisation (the descent is a serial data-dependent
   pointer-chase — there is no SIMD to emit) and **not** branch prediction.

### Layout

3. **The cache-engineered layouts (ILP / Blocked / Branchless) are net regressions, and worsen under
   load.** At 200 trees they are 1.4–1.6× slower than `Flat` single-thread, and the gap *widens* to
   ~2.4× at 8 threads (e.g. `BlockedIlpDfsFloat` 401k vs `Flat` 166k µs/op). They process multiple
   instances per node / touch more cache lines per step, so they are **memory-bandwidth-bound** and do
   not scale — exactly the FasterMolecularSurface lessons 12–13 ("reducing the wrong thing; the box is
   bandwidth-bound, not GC-bound"). The simple flat layouts (`Flat`, `ContiguousDfs`, `ShortLegacy`,
   `SeparateArraysBfs`) win and scale.

4. **`Flat` (and `ShortLegacy`/`ContiguousDfs`) is the right default for inference throughput.** It is
   fastest or near-fastest on Graal at every batch size and thread count measured, and scales to ~3.35 M
   predictions/s on 8 threads. `ShortLegacy`/`SuperShortLegacy` add ~50% memory savings (float arrays)
   at near-identical speed — and they are in the *legacy* (faithful) family (see PREDICTION-SEMANTICS).

### Native / SIMD

5. **Native (Panama FFM) does not beat the best Java here, and AVX2 is the *worst* native tier.**
   Legacy harness (200 trees, lower is faster, ms per 100 batches): scalar `NativePanama` ≈ 2,743 /
   `NativeFloatPanama` ≈ 2,775 land **mid-pack**, behind `ShortLegacy` (2,578) and `Flat` (2,597);
   `NativePanamaAvx2` ≈ 3,241 and `NativeFloatPanamaAvx2` ≈ 3,965 are **slower than their own scalar
   versions**; zero-copy off-heap barely helps (the FFM call boundary is not the bottleneck at this
   size). **Hypothesis (unverified here):** AVX-512/AVX2 license-downclock + short-burst scalar↔vector
   transition penalties — precisely FasterMolecularSurface's lesson 2 ("widest is not fastest"; pinning
   to 256-bit fixed it there). **Action before trusting AVX2:** measure the native path against a
   narrower/scalar variant; do not assume wider SIMD wins on a branchy, short-burst tree kernel.

### Batch size & metric

6. **Batch amortisation is real — sweep batch size, and pick the metric that matches the use case.**
   10× the rows costs only ~5.8× the time for `Flat` (fixed per-batch overhead amortises). Single-
   *prediction* latency, by contrast, is ~1.9 µs and **compiler-stable and contention-free** (8 threads
   don't slow it — the working set is one path per tree, L1-resident). Throughput-bound batch serving and
   latency-bound single calls are different regimes; report both (`ForestPredictBenchmark` does:
   `batchPredict` µs/op and `singlePredict` ns/op, with a `batchSize` param).

### Methodology

7. **JMH forks per benchmark; the legacy harness shares one JVM — and it shows.** The two harnesses
   agree to ~3% on 14 of 15 forests. The one divergence is `Original`: the legacy harness over-rates it
   because, running all forests in one long-lived JVM, the JIT has already profiled the hot `predict`
   path on other layouts before `Original` is measured. JMH's fresh fork per benchmark gives the
   trustworthy number. Use the legacy harness for quick relative sanity checks only; rank with JMH;
   always compare **ratios on the same JVM**, never absolute ms across runs.

8. **Measure, do not theorise.** Two confident hypotheses fell to short experiments this round: (a) "the
   Graal/C2 gap is branch misprediction" — refuted by the varied/uniform experiment; (b) "the shipped
   model's leaves are normalised so the score/legacy split is moot" — refuted by deserialising the model
   and measuring leaf sums (mean 1.51, 66% ≠ 1). The benchmark and the experiment decided, not intuition.

## Explicitly not worth doing (closed, with reason)

- **The ILP / Blocked / Branchless layouts** as production inference paths — net regressions that get
  worse under concurrency (lesson 3). Keep them only as measured baselines.
- **Assuming AVX2 > scalar native** — measured slower here (lesson 5); the wider path needs a downclock
  check first.
- **Reporting absolute milliseconds across runs / across harnesses** — turbo, load, and warm JIT make
  cross-run absolutes unreliable; the two harnesses are not interchangeable (lesson 7).
- **Ranking forests on HotSpot and shipping on GraalVM** (or vice-versa) — the winner inverts (lesson 1).

## Open ideas (not yet done)

- **Verify the AVX2 downclock hypothesis** (lesson 5): instrument the native path, and try a 256-bit /
  scalar native variant à la FasterMolecularSurface lesson 2.
- **Quantify the score-vs-legacy prediction delta** on the p2rank model (ties to PREDICTION-SEMANTICS):
  we measured the leaf sums (mean 1.51) but not the resulting end-to-end probability divergence.

*Done since first draft:* GraalVM is now in CI alongside HotSpot (the deployment JIT is validated), and
a pinned golden prediction baseline (`PredictionGoldenTest`) anchors the hot path in the standard suite.
