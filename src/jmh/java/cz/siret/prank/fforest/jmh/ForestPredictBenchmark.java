package cz.siret.prank.fforest.jmh;

import cz.siret.prank.fforest.FasterForest;
import cz.siret.prank.fforest.api.BinaryForest;
import cz.siret.prank.fforest.api.FasterForestConverter;
import cz.siret.prank.fforest.api.FasterForestConverter.ForestType;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * JMH prediction-speed microbenchmark for the heap-based forest representations.
 *
 * This is the JMH counterpart of the hand-rolled {@code PredictionSpeedBenchmark}
 * (which remains as a separate, coarse wall-clock harness). JMH handles JIT
 * warmup, dead-code elimination (returned values are consumed by the implicit
 * blackhole), and forking automatically.
 *
 * Run all heap forests:
 *   ./gradlew jmh
 *
 * Run a single representation, more forks:
 *   ./gradlew jmh -PjmhArgs="ForestPredict -p forestType=Flat -f 3"
 *
 * Opt in to a native variant (requires the native lib to be available):
 *   ./gradlew jmh -PjmhArgs="ForestPredict -p forestType=NativePanama"
 *
 * Forest types are converted lazily from a single trained {@link FasterForest}
 * (see {@link BenchData}), so retraining does not pollute the measurements.
 */
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
public class ForestPredictBenchmark {

    /**
     * Default set covers the portable, always-available heap representations.
     * Native variants ({@code NativePanama}, {@code NativePanamaAvx2},
     * {@code NativeFloatPanama}, {@code NativeFloatPanamaAvx2}) are supported by
     * {@link #buildForest} and can be selected explicitly with {@code -p forestType=...}.
     */
    @Param({"Original", "Flat", "LegacyFlat", "ShortLegacy",
            "InterleavedBfs", "InterleavedBfsDouble", "ContiguousBfsDouble",
            "SeparateArraysBfs", "BranchlessBfs", "ContiguousDfs",
            "IlpDfs", "BlockedIlpDfs", "IlpDfsFloat", "BlockedIlpDfsFloat",
            "FlatFloat"})
    public String forestType;

    @Param({"200"})
    public int numTrees;

    @Param({"0"})
    public int treeDepth;

    /**
     * Batch row count, tiled from the real dataset (~6950 rows). Affects
     * {@link #batchPredict} only — {@link #singlePredict} always uses one row.
     * Sweep with e.g. {@code -p batchSize=6950,69500,695000}; when sweeping,
     * narrow to the batch method to avoid redundant single-predict runs:
     * {@code -PjmhArgs="ForestPredictBenchmark.batchPredict -p batchSize=..."}.
     */
    @Param({"6950"})
    public int batchSize;

    private double[][] instances;
    private double[] singleInstance;
    private BinaryForest forest;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        FasterForest ff = BenchData.train(numTrees, treeDepth);
        instances = BenchData.instances(batchSize);
        singleInstance = instances[0];
        forest = buildForest(forestType, ff);
    }

    private static BinaryForest buildForest(String type, FasterForest ff) {
        switch (type) {
            case "Original":              return ff;
            case "Flat":                  return conv(ff, ForestType.FlatBinaryForest);
            case "LegacyFlat":            return conv(ff, ForestType.LegacyFlatBinaryForest);
            case "ShortLegacy":           return conv(ff, ForestType.ShortFlatBinaryForest);
            case "InterleavedBfs":        return conv(ff, ForestType.InterleavedBfsForest);
            case "InterleavedBfsDouble":  return conv(ff, ForestType.InterleavedBfsDoubleForest);
            case "ContiguousBfsDouble":   return conv(ff, ForestType.ContiguousBfsDoubleForest);
            case "SeparateArraysBfs":     return conv(ff, ForestType.SeparateArraysBfsForest);
            case "BranchlessBfs":         return conv(ff, ForestType.BranchlessBfsForest);
            case "ContiguousDfs":         return conv(ff, ForestType.ContiguousDfsForest);
            case "IlpDfs":                return conv(ff, ForestType.IlpDfsForest);
            case "BlockedIlpDfs":         return conv(ff, ForestType.BlockedIlpDfsForest);
            case "IlpDfsFloat":           return conv(ff, ForestType.IlpDfsFloatForest);
            case "BlockedIlpDfsFloat":    return conv(ff, ForestType.BlockedIlpDfsFloatForest);
            case "FlatFloat":             return conv(ff, ForestType.FlatBinaryFloatForest);
            case "NativePanama":          return conv(ff, ForestType.NativePanamaForest);
            case "NativePanamaAvx2":      return conv(ff, ForestType.NativePanamaForestAvx2);
            case "NativeFloatPanama":     return conv(ff, ForestType.NativePanamaFloatForest);
            case "NativeFloatPanamaAvx2": return conv(ff, ForestType.NativePanamaFloatForestAvx2);
            default:
                throw new IllegalArgumentException("Unknown forestType: " + type);
        }
    }

    private static BinaryForest conv(FasterForest ff, ForestType t) {
        return FasterForestConverter.convertFasterForest(ff, t);
    }

    /** Whole-dataset batch prediction (the primary production path). */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public double[] batchPredict() {
        return forest.predictForBatch(instances);
    }

    /** Single-instance prediction latency. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public double singlePredict() {
        return forest.predict(singleInstance);
    }
}
