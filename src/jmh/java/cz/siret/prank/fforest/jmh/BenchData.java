package cz.siret.prank.fforest.jmh;

import cz.siret.prank.fforest.FasterForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Shared dataset loading and forest training for the JMH benchmarks.
 *
 * Training a 200-tree forest is expensive, so the trained {@link FasterForest}
 * and the materialized instance array are cached per JVM fork and reused across
 * all @Param combinations that share the same (numTrees, treeDepth).
 */
final class BenchData {

    static final String DATA_FILE = "src/test/resources/data/p2rank-train.arff.gz";

    private static FasterForest cachedFF;
    private static double[][] cachedInstances;
    private static int cachedTrees = -1;
    private static int cachedDepth = -1;

    private BenchData() {}

    static synchronized FasterForest train(int numTrees, int treeDepth) throws Exception {
        if (cachedFF != null && cachedTrees == numTrees && cachedDepth == treeDepth) {
            return cachedFF;
        }

        Instances data = new ConverterUtils.DataSource(DATA_FILE).getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        FasterForest ff = new FasterForest();
        ff.setNumTrees(numTrees);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(treeDepth);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.buildClassifier(data);

        double[][] arr = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            arr[i] = data.get(i).toDoubleArray();
        }

        cachedFF = ff;
        cachedInstances = arr;
        cachedTrees = numTrees;
        cachedDepth = treeDepth;
        return ff;
    }

    static double[][] instances() {
        return cachedInstances;
    }

    /**
     * Returns the instance matrix sized to {@code targetSize} rows by tiling the
     * real dataset (row references are shared, not deep-copied). Larger sizes test
     * batch-throughput scaling and native call-overhead amortization, not data
     * diversity — the unique rows remain those of the underlying dataset.
     *
     * @param targetSize desired row count; {@code <= 0} means "use the dataset as-is"
     */
    static double[][] instances(int targetSize) {
        double[][] base = cachedInstances;
        if (targetSize <= 0 || targetSize == base.length) {
            return base;
        }
        double[][] out = new double[targetSize][];
        for (int i = 0; i < targetSize; i++) {
            out[i] = base[i % base.length];
        }
        return out;
    }
}
