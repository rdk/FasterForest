package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.*;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.*;

/**
 * Prediction speed benchmark for all forest representations.
 *
 * Compares prediction throughput across forest types:
 * - FasterForest (original tree-based)
 * - FlatBinaryForest (new flat layout)
 * - LegacyFlatBinaryForest (flattened arrays)
 * - LegacyFlatBinaryForest optimized (reordered by access counts)
 * - ShortLegacyFlatBinaryForest (compact float/short arrays)
 * - SuperShortLegacyFlatBinaryForest (ultra-compact)
 * - InterleavedBfsForest (cache-optimized BFS layout)
 *
 * Skipped by default. Run with:
 *   ./gradlew benchmark
 * or:
 *   ./gradlew test --tests "*.PredictionSpeedBenchmark.benchmarkAll" -Dbenchmark=true -i
 */
public class PredictionSpeedBenchmark {

    static final String DATA_DIR = "src/test/resources/data/";

    // --- toggle individual implementations on/off ---

    static final boolean ENABLE_ORIGINAL_FF       = true;
    static final boolean ENABLE_FLAT               = true;
    static final boolean ENABLE_LEGACY_FLAT        = true;
    static final boolean ENABLE_LEGACY_FLAT_OPT    = false;
    static final boolean ENABLE_SHORT_LEGACY       = true;
    static final boolean ENABLE_SUPER_SHORT        = false;
    static final boolean ENABLE_INTERLEAVED_BFS    = true;
    static final boolean ENABLE_INTERLEAVED_BFS_D  = true;
    static final boolean ENABLE_CONTIGUOUS_BFS_D   = true;
    static final boolean ENABLE_SEPARATE_BFS       = true;
    static final boolean ENABLE_BRANCHLESS_BFS     = true;
    static final boolean ENABLE_CONTIGUOUS_DFS     = true;
    static final boolean ENABLE_ILP_DFS            = true;
    static final boolean ENABLE_BLOCKED_ILP_DFS    = true;
    static final boolean ENABLE_ILP_DFS_FLOAT      = true;
    static final boolean ENABLE_BLOCKED_ILP_DFS_FLOAT = true;
    static final boolean ENABLE_FLAT_FLOAT         = true;
    static final boolean ENABLE_NATIVE_PANAMA      = true;
    static final boolean ENABLE_NATIVE_PANAMA_0COPY = true;
    static final boolean ENABLE_NATIVE_PANAMA_AVX2 = true;
    static final boolean ENABLE_NATIVE_PANAMA_AVX2_0COPY = true;
    static final boolean ENABLE_NATIVE_FLOAT_PANAMA      = true;
    static final boolean ENABLE_NATIVE_FLOAT_PANAMA_0COPY = true;
    static final boolean ENABLE_NATIVE_FLOAT_PANAMA_AVX2 = true;
    static final boolean ENABLE_NATIVE_FLOAT_PANAMA_AVX2_0COPY = true;

    // --- benchmark parameters (overridable via -D system properties) ---

    static final int NUM_TREES = Integer.getInteger("bench.numTrees", 200);
    static final int TREE_DEPTH = Integer.getInteger("bench.treeDepth", 0);
    static final int WARMUP_ROUNDS = 1;
    static final int MEASURE_ROUNDS = Integer.getInteger("bench.measureRounds", 5);
    static final int ITERS_PER_ROUND = Integer.getInteger("bench.itersPerRound", 100);
    // Batch row count; rows are tiled from the real dataset. 0 = use dataset as-is.
    static final int BATCH_SIZE = Integer.getInteger("bench.batchSize", 0);

    Instances dataset;
    double[][] instances;

    // --- forests ---

    FasterForest ff;
    BinaryForest flatNewForest;
    LegacyFlatBinaryForest legacyFlatForest;
    LegacyFlatBinaryForest optimizedForest;
    ShortLegacyFlatBinaryForest shortForest;
    SuperShortLegacyFlatBinaryForest superShortForest;
    BinaryForest interleavedBfsForest;
    BinaryForest interleavedBfsDoubleForest;
    BinaryForest contiguousBfsDoubleForest;
    BinaryForest separateArraysBfsForest;
    BinaryForest branchlessBfsForest;
    BinaryForest contiguousDfsForest;
    BinaryForest ilpDfsForest;
    BinaryForest blockedIlpDfsForest;
    BinaryForest ilpDfsFloatForest;
    BinaryForest blockedIlpDfsFloatForest;
    BinaryForest flatFloatForest;
    BinaryForest nativePanamaForest;
    BinaryForest nativePanamaAvx2Forest;
    BinaryForest nativeFloatPanamaForest;
    BinaryForest nativeFloatPanamaAvx2Forest;

    // --- zero-copy native data ---
    Arena offHeapArena;
    MemorySegment offHeapInstances;
    Arena offHeapArenaAvx2;
    MemorySegment offHeapInstancesAvx2;
    Arena offHeapArenaFloat;
    MemorySegment offHeapInstancesFloat;
    Arena offHeapArenaFloatAvx2;
    MemorySegment offHeapInstancesFloatAvx2;

    // =========================================================================

    private static Instances loadDataset(String path) throws Exception {
        Instances data = new ConverterUtils.DataSource(path).getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    @Before
    public void init() throws Exception {
        Assume.assumeTrue("Benchmark skipped. Run with -Dbenchmark=true or ./gradlew benchmark",
                Boolean.getBoolean("benchmark"));

        dataset = loadDataset(DATA_DIR + "p2rank-train.arff.gz");
        instances = tileToSize(instancesToArrays(dataset), BATCH_SIZE);

        ff = new FasterForest();
        ff.setNumTrees(NUM_TREES);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(TREE_DEPTH);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);

        System.out.printf("Training FasterForest with %d trees...%n", NUM_TREES);
        long t0 = System.nanoTime();
        ff.buildClassifier(dataset);
        long trainMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("Training done in %d ms%n", trainMs);

        if (ENABLE_FLAT) {
            flatNewForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.FlatBinaryForest);
        }

        if (ENABLE_LEGACY_FLAT || ENABLE_LEGACY_FLAT_OPT || ENABLE_SHORT_LEGACY || ENABLE_SUPER_SHORT) {
            legacyFlatForest = ff.toFlatBinaryForest();
        }

        if (ENABLE_LEGACY_FLAT_OPT && legacyFlatForest != null) {
            OptimizingFlatBinaryForest optimizingForest = new OptimizingFlatBinaryForest(legacyFlatForest);
            for (Instance inst : dataset) {
                optimizingForest.predict(inst.toDoubleArray());
            }
            optimizedForest = optimizingForest.buildOptimizedForest();
        }

        if (ENABLE_SHORT_LEGACY && legacyFlatForest != null) {
            shortForest = ShortLegacyFlatBinaryForest.from(legacyFlatForest);
        }
        if (ENABLE_SUPER_SHORT && legacyFlatForest != null) {
            superShortForest = SuperShortLegacyFlatBinaryForest.from(legacyFlatForest);
        }
        if (ENABLE_INTERLEAVED_BFS) {
            interleavedBfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.InterleavedBfsForest);
        }
        if (ENABLE_INTERLEAVED_BFS_D) {
            interleavedBfsDoubleForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.InterleavedBfsDoubleForest);
        }
        if (ENABLE_CONTIGUOUS_BFS_D) {
            contiguousBfsDoubleForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.ContiguousBfsDoubleForest);
        }
        if (ENABLE_SEPARATE_BFS) {
            separateArraysBfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.SeparateArraysBfsForest);
        }
        if (ENABLE_BRANCHLESS_BFS) {
            branchlessBfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.BranchlessBfsForest);
        }
        if (ENABLE_CONTIGUOUS_DFS) {
            contiguousDfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.ContiguousDfsForest);
        }
        if (ENABLE_ILP_DFS) {
            ilpDfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.IlpDfsForest);
        }
        if (ENABLE_BLOCKED_ILP_DFS) {
            blockedIlpDfsForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.BlockedIlpDfsForest);
        }
        if (ENABLE_ILP_DFS_FLOAT) {
            ilpDfsFloatForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.IlpDfsFloatForest);
        }
        if (ENABLE_BLOCKED_ILP_DFS_FLOAT) {
            blockedIlpDfsFloatForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.BlockedIlpDfsFloatForest);
        }
        if (ENABLE_FLAT_FLOAT) {
            flatFloatForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.FlatBinaryFloatForest);
        }
        if ((ENABLE_NATIVE_PANAMA || ENABLE_NATIVE_PANAMA_0COPY) && NativePanamaForest.isAvailable()) {
            nativePanamaForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.NativePanamaForest);
            System.out.printf("Native SIMD level: %d%n", NativePanamaForest.simdLevel());
        }
        if (ENABLE_NATIVE_PANAMA_0COPY && nativePanamaForest != null) {
            offHeapArena = Arena.ofShared();
            offHeapInstances = NativePanamaForest.flattenToOffHeap(
                    instances, nativePanamaForest.getNumAttributes(), offHeapArena);
        }
        if ((ENABLE_NATIVE_PANAMA_AVX2 || ENABLE_NATIVE_PANAMA_AVX2_0COPY) && NativePanamaForestAvx2.isAvx2Available()) {
            nativePanamaAvx2Forest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.NativePanamaForestAvx2);
        }
        if (ENABLE_NATIVE_PANAMA_AVX2_0COPY && nativePanamaAvx2Forest != null) {
            offHeapArenaAvx2 = Arena.ofShared();
            offHeapInstancesAvx2 = NativePanamaForest.flattenToOffHeap(
                    instances, nativePanamaAvx2Forest.getNumAttributes(), offHeapArenaAvx2);
        }
        if ((ENABLE_NATIVE_FLOAT_PANAMA || ENABLE_NATIVE_FLOAT_PANAMA_0COPY) && NativePanamaFloatForest.isAvailable()) {
            nativeFloatPanamaForest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.NativePanamaFloatForest);
        }
        if (ENABLE_NATIVE_FLOAT_PANAMA_0COPY && nativeFloatPanamaForest != null) {
            offHeapArenaFloat = Arena.ofShared();
            offHeapInstancesFloat = NativePanamaForest.flattenToOffHeap(
                    instances, nativeFloatPanamaForest.getNumAttributes(), offHeapArenaFloat);
        }
        if ((ENABLE_NATIVE_FLOAT_PANAMA_AVX2 || ENABLE_NATIVE_FLOAT_PANAMA_AVX2_0COPY) && NativePanamaFloatForestAvx2.isAvx2Available()) {
            nativeFloatPanamaAvx2Forest = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.NativePanamaFloatForestAvx2);
        }
        if (ENABLE_NATIVE_FLOAT_PANAMA_AVX2_0COPY && nativeFloatPanamaAvx2Forest != null) {
            offHeapArenaFloatAvx2 = Arena.ofShared();
            offHeapInstancesFloatAvx2 = NativePanamaForest.flattenToOffHeap(
                    instances, nativeFloatPanamaAvx2Forest.getNumAttributes(), offHeapArenaFloatAvx2);
        }

        System.out.printf("Dataset: %d instances, %d attributes%n", dataset.size(), dataset.numAttributes() - 1);
        System.out.printf("Forest: %d trees, max depth %d%n", ff.getNumTrees(), ff.calculateMaxTreeDepth());
        System.out.println();
    }

    // =========================================================================
    //  Benchmarks
    // =========================================================================

    //@Test
    public void benchmarkSinglePrediction() throws Exception {
        System.out.println("=== Single Prediction Benchmark ===");
        printConfig();
        printHeader();

        for (Map.Entry<String, BinaryForest> entry : buildForestMap().entrySet()) {
            printResult(entry.getKey(), runBenchmark(entry.getValue(), false));
        }
    }

    //@Test
    public void benchmarkBatchPrediction() throws Exception {
        System.out.println("=== Batch Prediction Benchmark ===");
        printConfig();
        printHeader();

        for (Map.Entry<String, BinaryForest> entry : buildForestMap().entrySet()) {
            printResult(entry.getKey(), runBenchmark(entry.getValue(), true));
        }
    }

    @Test
    public void benchmarkAll() throws Exception {
        printConfig();
        LinkedHashMap<String, BinaryForest> forests = buildForestMap();

        // Run benchmarks and collect results
        LinkedHashMap<String, BenchResult> results = new LinkedHashMap<>();
        for (Map.Entry<String, BinaryForest> entry : forests.entrySet()) {
            System.out.printf("Running: %s ...%n", entry.getKey());
            results.put(entry.getKey(), runBenchmark(entry.getValue(), true));
        }

        // Zero-copy native benchmarks (special case — uses pre-flattened off-heap data)
        if (ENABLE_NATIVE_PANAMA_0COPY && nativePanamaForest != null) {
            System.out.printf("Running: %s ...%n", "NativePanama0copy");
            results.put("NativePanama0copy", runNativeZeroCopyBenchmark(
                    (NativePanamaForest) nativePanamaForest, offHeapInstances));
        }
        if (ENABLE_NATIVE_PANAMA_AVX2_0COPY && nativePanamaAvx2Forest != null) {
            System.out.printf("Running: %s ...%n", "NativePanamaAvx20copy");
            results.put("NativePanamaAvx20copy", runNativeZeroCopyBenchmark(
                    (NativePanamaForest) nativePanamaAvx2Forest, offHeapInstancesAvx2));
        }
        if (ENABLE_NATIVE_FLOAT_PANAMA_0COPY && nativeFloatPanamaForest != null) {
            System.out.printf("Running: %s ...%n", "NativeFloatPanama0copy");
            results.put("NativeFloatPanama0copy", runNativeFloatZeroCopyBenchmark(
                    (NativePanamaFloatForest) nativeFloatPanamaForest, offHeapInstancesFloat));
        }
        if (ENABLE_NATIVE_FLOAT_PANAMA_AVX2_0COPY && nativeFloatPanamaAvx2Forest != null) {
            System.out.printf("Running: %s ...%n", "NativeFloatPanamaAvx20copy");
            results.put("NativeFloatPanamaAvx20copy", runNativeFloatZeroCopyBenchmark(
                    (NativePanamaFloatForest) nativeFloatPanamaAvx2Forest, offHeapInstancesFloatAvx2));
        }

        // JSON output
        System.out.println();
        System.out.println("=== JSON ===");
        System.out.println("[");
        int i = 0;
        for (Map.Entry<String, BenchResult> entry : results.entrySet()) {
            BenchResult r = entry.getValue();
            System.out.printf("  {\"name\": \"%s\", \"mean_ms\": %.1f, \"std_ms\": %.1f, \"min_ms\": %d, \"max_ms\": %d, \"pred_per_sec\": %.0f}%s%n",
                    entry.getKey(), r.mean(), r.stddev(), r.min(), r.max(), r.predictionsPerSecond(),
                    (++i < results.size()) ? "," : "");
        }
        System.out.println("]");

        // Formatted table at the end
        System.out.println();
        System.out.println("=== Batch Prediction Benchmark ===");
        System.out.println();
        printHeader();
        for (Map.Entry<String, BenchResult> entry : results.entrySet()) {
            printResult(entry.getKey(), entry.getValue());
        }
    }

    // =========================================================================
    //  Internals
    // =========================================================================

    private LinkedHashMap<String, BinaryForest> buildForestMap() {
        LinkedHashMap<String, BinaryForest> forests = new LinkedHashMap<>();
        if (ENABLE_ORIGINAL_FF)    forests.put("Original (FF)", ff);
        if (ENABLE_FLAT)           forests.put("Flat", flatNewForest);
        if (ENABLE_LEGACY_FLAT)    forests.put("LegacyFlat", legacyFlatForest);
        if (ENABLE_LEGACY_FLAT_OPT) forests.put("LegacyFlat optimized", optimizedForest);
        if (ENABLE_SHORT_LEGACY)   forests.put("ShortLegacy", shortForest);
        if (ENABLE_SUPER_SHORT)    forests.put("SuperShort", superShortForest);
        if (ENABLE_INTERLEAVED_BFS) forests.put("InterleavedBfs", interleavedBfsForest);
        if (ENABLE_INTERLEAVED_BFS_D) forests.put("InterleavedBfsDouble", interleavedBfsDoubleForest);
        if (ENABLE_CONTIGUOUS_BFS_D) forests.put("ContiguousBfsDouble", contiguousBfsDoubleForest);
        if (ENABLE_SEPARATE_BFS) forests.put("SeparateArraysBfs", separateArraysBfsForest);
        if (ENABLE_BRANCHLESS_BFS) forests.put("BranchlessBfs", branchlessBfsForest);
        if (ENABLE_CONTIGUOUS_DFS) forests.put("ContiguousDfs", contiguousDfsForest);
        if (ENABLE_ILP_DFS) forests.put("IlpDfs", ilpDfsForest);
        if (ENABLE_BLOCKED_ILP_DFS) forests.put("BlockedIlpDfs", blockedIlpDfsForest);
        if (ENABLE_ILP_DFS_FLOAT) forests.put("IlpDfsFloat", ilpDfsFloatForest);
        if (ENABLE_BLOCKED_ILP_DFS_FLOAT) forests.put("BlockedIlpDfsFloat", blockedIlpDfsFloatForest);
        if (ENABLE_FLAT_FLOAT) forests.put("FlatFloat", flatFloatForest);
        if (ENABLE_NATIVE_PANAMA && nativePanamaForest != null) forests.put("NativePanama", nativePanamaForest);
        if (ENABLE_NATIVE_PANAMA_AVX2 && nativePanamaAvx2Forest != null) forests.put("NativePanamaAvx2", nativePanamaAvx2Forest);
        if (ENABLE_NATIVE_FLOAT_PANAMA && nativeFloatPanamaForest != null) forests.put("NativeFloatPanama", nativeFloatPanamaForest);
        if (ENABLE_NATIVE_FLOAT_PANAMA_AVX2 && nativeFloatPanamaAvx2Forest != null) forests.put("NativeFloatPanamaAvx2", nativeFloatPanamaAvx2Forest);
        return forests;
    }

    private BenchResult runBenchmark(BinaryForest forest, boolean batch) {
        for (int w = 0; w < WARMUP_ROUNDS; w++) {
            runOneRound(forest, batch);
            System.gc();
        }

        long[] times = new long[MEASURE_ROUNDS];
        for (int r = 0; r < MEASURE_ROUNDS; r++) {
            times[r] = runOneRound(forest, batch);
            System.gc();
        }

        return new BenchResult(times, (long) ITERS_PER_ROUND * instances.length);
    }

    private BenchResult runNativeFloatZeroCopyBenchmark(NativePanamaFloatForest npf, MemorySegment data) {
        int n = instances.length;

        for (int w = 0; w < WARMUP_ROUNDS; w++) {
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                npf.predictForBatchContiguous(data, n);
            }
            System.gc();
        }

        long[] times = new long[MEASURE_ROUNDS];
        for (int r = 0; r < MEASURE_ROUNDS; r++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                npf.predictForBatchContiguous(data, n);
            }
            times[r] = (System.nanoTime() - t0) / 1_000_000;
            System.gc();
        }

        return new BenchResult(times, (long) ITERS_PER_ROUND * n);
    }

    private BenchResult runNativeZeroCopyBenchmark(NativePanamaForest npf, MemorySegment data) {
        int n = instances.length;

        for (int w = 0; w < WARMUP_ROUNDS; w++) {
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                npf.predictForBatchContiguous(data, n);
            }
            System.gc();
        }

        long[] times = new long[MEASURE_ROUNDS];
        for (int r = 0; r < MEASURE_ROUNDS; r++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                npf.predictForBatchContiguous(data, n);
            }
            times[r] = (System.nanoTime() - t0) / 1_000_000;
            System.gc();
        }

        return new BenchResult(times, (long) ITERS_PER_ROUND * n);
    }

    private long runOneRound(BinaryForest forest, boolean batch) {
        long t0 = System.nanoTime();
        if (batch) {
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                forest.predictForBatch(instances);
            }
        } else {
            for (int i = 0; i < ITERS_PER_ROUND; i++) {
                for (double[] inst : instances) {
                    forest.predict(inst);
                }
            }
        }
        return (System.nanoTime() - t0) / 1_000_000;
    }

    private void printConfig() {
        System.out.printf("Config: warmup=%d, measured=%d, iters=%d, instances=%d%n",
                WARMUP_ROUNDS, MEASURE_ROUNDS, ITERS_PER_ROUND, instances.length);
        System.out.printf("Total predictions per round: %,d%n", (long) ITERS_PER_ROUND * instances.length);
        System.out.println();
    }

    private void printHeader() {
        System.out.printf("%-25s %8s %8s %8s %8s %12s%n",
                "Forest", "Mean ms", "Std ms", "Min ms", "Max ms", "Pred/sec");
        System.out.println("-".repeat(82));
    }

    private void printResult(String name, BenchResult r) {
        System.out.printf("%-25s %8.1f %8.1f %8d %8d %,12.0f%n",
                name, r.mean(), r.stddev(), r.min(), r.max(), r.predictionsPerSecond());
    }

    private double[][] instancesToArrays(Instances data) {
        double[][] result = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            result[i] = data.get(i).toDoubleArray();
        }
        return result;
    }

    /**
     * Tiles {@code base} to {@code targetSize} rows by repeating rows (references
     * shared, not copied). {@code targetSize <= 0} returns {@code base} unchanged.
     */
    private static double[][] tileToSize(double[][] base, int targetSize) {
        if (targetSize <= 0 || targetSize == base.length) {
            return base;
        }
        double[][] out = new double[targetSize][];
        for (int i = 0; i < targetSize; i++) {
            out[i] = base[i % base.length];
        }
        return out;
    }

    // =========================================================================

    static class BenchResult {
        final long[] timesMs;
        final long predictionsPerRound;

        BenchResult(long[] timesMs, long predictionsPerRound) {
            this.timesMs = timesMs;
            this.predictionsPerRound = predictionsPerRound;
        }

        double mean() {
            long sum = 0;
            for (long t : timesMs) sum += t;
            return (double) sum / timesMs.length;
        }

        double stddev() {
            double m = mean();
            double sumSq = 0;
            for (long t : timesMs) sumSq += (t - m) * (t - m);
            return Math.sqrt(sumSq / timesMs.length);
        }

        long min() {
            long min = Long.MAX_VALUE;
            for (long t : timesMs) if (t < min) min = t;
            return min;
        }

        long max() {
            long max = Long.MIN_VALUE;
            for (long t : timesMs) if (t > max) max = t;
            return max;
        }

        double predictionsPerSecond() {
            double meanSec = mean() / 1000.0;
            if (meanSec == 0) return Double.POSITIVE_INFINITY;
            return predictionsPerRound / meanSec;
        }
    }

}
