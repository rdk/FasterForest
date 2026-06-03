# FasterForest — conventions for agents & contributors

This file is the definition-of-done for work in this repo. It is auto-loaded by Claude Code; other
agents should read it first. Keep it short — it points at the canonical docs rather than duplicating
them.

## Invariants (do not break)

1. **Never change the default prediction semantics.** The shipped/default path is the *faithful*
   (legacy, sum-then-normalize) family, which reproduces the trained model exactly. Optimizations that
   change the floating-point result ship only as separate, opt-in variants with their own tolerance
   tests — never as the default. See [PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md).
2. **Leaves generally do not sum to 1** (they encode mean bootstrap multiplicity, ~1.5 on real models).
   Code that assumes normalized leaves is wrong. This is *why* the two prediction families diverge.
3. **The deployment JVM is GraalVM.** Performance rankings invert between Graal and HotSpot C2, so a
   result is only meaningful with its JVM stated. Benchmark on GraalVM; CI runs both.
4. **Bit-exact equivalence is a contract.** Every inference variant is validated against a reference of
   its family. Build/keep the oracle; optimize fearlessly behind it.
5. **Don't break serialization of the persisted forests.** Trained models are Java-serialized and shipped
   (p2rank distributes its default as a `LegacyFlatBinaryForest`). Keep a stable class name,
   `serialVersionUID`, and field layout for `LegacyFlatBinaryForest` / `FlatBinaryForest`,
   `FasterForest` / `FasterForest2`, and their object graph (`FastRfBagging`, `FasterTree`). Adding a
   field is OK (old streams default it); renaming the class or removing/retyping a serialized field is
   not. (This is why we *kept* the misleading `LegacyFlatBinaryForest` name instead of renaming it.)

## Single sources of truth (update the owner, not a copy)

| Topic | Canonical doc |
|---|---|
| which variant to use / variant status | [VARIANTS.md](VARIANTS.md) |
| prediction correctness (score vs legacy) | [PREDICTION-SEMANTICS.md](PREDICTION-SEMANTICS.md) |
| speed claims + benchmarking method | [PERFORMANCE-LESSONS.md](PERFORMANCE-LESSONS.md) |
| architecture / how-to | [DEVELOPMENT.md](DEVELOPMENT.md) |

**A perf number without a JVM + harness + date does not go in the README** — it goes in
PERFORMANCE-LESSONS.md. The README links; it does not own numbers.

## Checklist — adding a forest variant

1. Implement `BinaryForest`; state its **family** (faithful/score) and a one-line rationale in the class
   Javadoc.
2. Add it to the cross-representation equivalence test (`BinaryForestInferenceTest`) at the
   family-appropriate tolerance. A variant without an equivalence test is not done.
3. Measure it on **GraalVM** with JMH (`./jmh.sh` / `ForestPredictBenchmark`), not just the legacy harness.
4. Add a row to [VARIANTS.md](VARIANTS.md) with its status. A net-slower variant is kept as a
   `regression` baseline with its rationale — **don't delete negative results, document them.**
5. Default selection (`FasterForestConverter`, p2rank's `rf_flatten_target`) must never auto-pick an
   `experimental`/`regression` variant.

## Checklist — making or revising a perf claim

- Compare **ratios on the same JVM**, never absolute ms across runs.
- Don't mix the JMH and legacy harnesses in one table (they are not interchangeable; the legacy harness
  shares one JVM and warm-profiles later forests).
- State JVM name+version, harness, and date. Put it in PERFORMANCE-LESSONS.md.
- Prefer JMH (forks per benchmark) for rankings; the legacy harness is for quick relative checks only.

## Methodology creed

**Measure, do not theorize.** Confident hypotheses have been wrong here (the AVX2 win, the
branch-misprediction explanation, "leaves are normalized") — a short experiment settled each. The sibling
repo `../FasterMolecularSurface/docs/performance-lessons.md` is the model for this discipline.

## Gotchas (this repo) — read before benchmarking/testing

These cost real time to rediscover; they are intentional, not bugs.

- **Test stdout is hidden.** The `test` task sets `showStandardStreams=false`, so `System.out` from a
  test prints nothing on the console. To see it, read `build/test-results/test/TEST-*.xml` (or run a
  throwaway via the `benchmark` task, which has streams on).
- **`-D` system properties are NOT forwarded to the test JVM** unless explicitly listed in `build.gradle`
  (only `benchmark`, `golden.regenerate`, and `ci.native.required` are). Add a `systemProperty` line if you need a new one.
- **`java22` output must precede `main` on the classpath**, or the Java 17 Panama *stubs* win and
  `NativePanamaForest.isAvailable()` silently returns false (native forests vanish). All tasks already
  do this; preserve it if you touch `build.gradle`. Never delete `src/main/java22/` — tests compile
  against it.
- **`-PjmhArgs` is whitespace-split**, then each token is passed to JMH. Multi-word JVM args must be a
  single token: `-jvmArgs=--enable-native-access=ALL-UNNAMED` (with `=`), never `-jvmArgs --enable-...`.
- **JMH holds a global `/tmp/jmh.lock`** — only one JMH run at a time on a machine. Run Graal and C2
  comparisons sequentially, not concurrently.
- **`perf` profiling is likely blocked** (`perf_event_paranoid` high, no hsdis) — so `-prof perfasm/
  perfnorm` won't read counters. Attribute codegen behaviourally (controlled experiments), as in
  PERFORMANCE-LESSONS.md.
- **Benchmark a specific JVM** while the build pins a toolchain by pointing JMH at it:
  `./gradlew jmh -PjmhArgs="ForestPredict -jvm /path/to/java -jvmArgs=--enable-native-access=ALL-UNNAMED"`.

## Git

- Commit/push only when asked. Branch first if on the default branch (`develop`).
- No `Co-Authored-By` lines in commit messages.
- When committing docs that cross-link, verify every linked file is tracked before pushing.
