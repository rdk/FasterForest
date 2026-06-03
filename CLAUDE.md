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

## Git

- Commit/push only when asked. Branch first if on the default branch (`develop`).
- No `Co-Authored-By` lines in commit messages.
