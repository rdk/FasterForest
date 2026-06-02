package cz.siret.prank.fforest.jmh;

import cz.siret.prank.fforest.FasterForest;
import cz.siret.prank.fforest.api.BinaryForest;
import cz.siret.prank.fforest.api.FasterForestConverter;
import cz.siret.prank.fforest.api.FasterForestConverter.ForestType;
import org.openjdk.jmh.annotations.*;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Investigation harness: is the Graal-vs-C2 gap on {@code Flat} driven by
 * branch misprediction in the data-dependent tree descent?
 *
 * The inner loop is {@code if (x[attr] < split) node = left; else node = right;}
 * — a hard-to-predict, data-dependent branch. If C2 compiles it as a real branch
 * and Graal as a branchless select (cmov), then:
 *
 *   - dataMode=uniform  : all rows identical → every tree takes ONE fixed path →
 *                         the branch is perfectly predictable. C2's branch cost
 *                         goes to ~0; Graal (already branchless) is unaffected.
 *   - dataMode=varied   : real rows → ~50% mispredict per node → C2 pays full price.
 *
 * Expected signature of the branch-misprediction hypothesis:
 *   C2:    varied  >>  uniform        (large speedup when predictable)
 *   Graal: varied  ~=  uniform        (little change — no branch to mispredict)
 *
 * Single-threaded on purpose: this isolates per-core codegen, not memory contention.
 * Run under each compiler with -jvm (see jmh.sh / the C2 invocation in the notes).
 */
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgsAppend = {"--enable-native-access=ALL-UNNAMED"})
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Threads(1)
public class BranchSensitivityBenchmark {

    @Param({"Flat", "Original"})
    public String forestType;

    @Param({"varied", "uniform"})
    public String dataMode;

    @Param({"69500"})
    public int batchSize;

    @Param({"200"})
    public int numTrees;

    private double[][] instances;
    private BinaryForest forest;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        FasterForest ff = BenchData.train(numTrees, 0);

        if ("uniform".equals(dataMode)) {
            double[] one = BenchData.instances()[0];
            instances = new double[batchSize][];
            Arrays.fill(instances, one);   // every traversal follows one fixed path
        } else {
            instances = BenchData.instances(batchSize);
        }

        forest = "Original".equals(forestType)
                ? ff
                : FasterForestConverter.convertFasterForest(ff, ForestType.FlatBinaryForest);
    }

    @Benchmark
    public double[] batchPredict() {
        return forest.predictForBatch(instances);
    }
}
