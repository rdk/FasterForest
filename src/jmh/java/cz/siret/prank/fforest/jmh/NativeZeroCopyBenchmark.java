package cz.siret.prank.fforest.jmh;

import cz.siret.prank.fforest.FasterForest;
import cz.siret.prank.fforest.api.FasterForestConverter;
import cz.siret.prank.fforest.api.FasterForestConverter.ForestType;
import cz.siret.prank.fforest.api.NativePanamaForest;
import cz.siret.prank.fforest.api.NativePanamaForestAvx2;
import cz.siret.prank.fforest.api.NativePanamaFloatForest;
import cz.siret.prank.fforest.api.NativePanamaFloatForestAvx2;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark for the native (Panama FFM) zero-copy prediction path.
 *
 * The instance matrix is flattened once into an off-heap {@link MemorySegment}
 * during setup; each invocation calls {@code predictForBatchContiguous} with no
 * per-call marshalling — this isolates the native compute cost from data copy.
 *
 * Requires the native library to be available. If a requested variant is not
 * available on the current machine, setup throws and JMH reports that parameter
 * combination as failed (the {@code jmh} Gradle task runs with {@code -foe false}
 * so the rest of the run continues).
 *
 *   ./gradlew jmh -PjmhArgs="NativeZeroCopy"
 */
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class NativeZeroCopyBenchmark {

    @Param({"NativePanama", "NativePanamaAvx2", "NativeFloatPanama", "NativeFloatPanamaAvx2"})
    public String forestType;

    @Param({"200"})
    public int numTrees;

    @Param({"0"})
    public int treeDepth;

    /** Batch row count, tiled from the real dataset. Larger sizes amortize the
     *  single native (FFM) call over more instances — sweep to find the crossover
     *  where native overtakes the JVM layouts: {@code -p batchSize=6950,69500,695000}. */
    @Param({"6950"})
    public int batchSize;

    private Arena arena;
    private MemorySegment offHeapData;
    private int numInstances;

    private NativePanamaForest doubleForest;
    private NativePanamaFloatForest floatForest;
    private boolean isFloat;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        if (!isAvailable(forestType)) {
            throw new IllegalStateException("Native variant not available: " + forestType);
        }

        FasterForest ff = BenchData.train(numTrees, treeDepth);
        double[][] instances = BenchData.instances(batchSize);
        numInstances = instances.length;

        int numAttributes;
        switch (forestType) {
            case "NativePanama":
                doubleForest = (NativePanamaForest) conv(ff, ForestType.NativePanamaForest);
                numAttributes = doubleForest.getNumAttributes();
                break;
            case "NativePanamaAvx2":
                doubleForest = (NativePanamaForest) conv(ff, ForestType.NativePanamaForestAvx2);
                numAttributes = doubleForest.getNumAttributes();
                break;
            case "NativeFloatPanama":
                floatForest = (NativePanamaFloatForest) conv(ff, ForestType.NativePanamaFloatForest);
                isFloat = true;
                numAttributes = floatForest.getNumAttributes();
                break;
            case "NativeFloatPanamaAvx2":
                floatForest = (NativePanamaFloatForest) conv(ff, ForestType.NativePanamaFloatForestAvx2);
                isFloat = true;
                numAttributes = floatForest.getNumAttributes();
                break;
            default:
                throw new IllegalArgumentException("Unknown forestType: " + forestType);
        }

        arena = Arena.ofShared();
        offHeapData = NativePanamaForest.flattenToOffHeap(instances, numAttributes, arena);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    private static boolean isAvailable(String type) {
        switch (type) {
            case "NativePanama":          return NativePanamaForest.isAvailable();
            case "NativePanamaAvx2":      return NativePanamaForestAvx2.isAvx2Available();
            case "NativeFloatPanama":     return NativePanamaFloatForest.isAvailable();
            case "NativeFloatPanamaAvx2": return NativePanamaFloatForestAvx2.isAvx2Available();
            default:                      return false;
        }
    }

    private static cz.siret.prank.fforest.api.BinaryForest conv(FasterForest ff, ForestType t) {
        return FasterForestConverter.convertFasterForest(ff, t);
    }

    @Benchmark
    public double[] batchPredictZeroCopy() {
        return isFloat
                ? floatForest.predictForBatchContiguous(offHeapData, numInstances)
                : doubleForest.predictForBatchContiguous(offHeapData, numInstances);
    }
}
