package cz.siret.prank.fforest;

import cz.siret.prank.fforest2.FasterForest2;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Verifies that training FasterForest and FasterForest2 is deterministic:
 * the same seed and parameters produce bit-identical trees and predictions,
 * regardless of thread count.
 */
public class TrainingDeterminismTest {

    static final String DATA_DIR = "src/test/resources/data/";

    Instances dataset;

    @Before
    public void init() throws Exception {
        dataset = loadDataset(DATA_DIR + "p2rank-train.arff.gz");
    }

    // =========================================================================
    //  FasterForest (v1)
    // =========================================================================

    @Test
    public void ff1_sameSeed_singleThread_producesSamePredictions() throws Exception {
        FasterForest ff1 = trainFF1(42, 32, 1);
        FasterForest ff2 = trainFF1(42, 32, 1);
        assertIdenticalForests("FF1 single-thread", ff1, ff2);
    }

    @Test
    public void ff1_sameSeed_multiThread_producesSamePredictions() throws Exception {
        FasterForest ff1 = trainFF1(42, 32, 4);
        FasterForest ff2 = trainFF1(42, 32, 4);
        assertIdenticalForests("FF1 multi-thread", ff1, ff2);
    }

    @Test
    public void ff1_differentThreadCount_producesSamePredictions() throws Exception {
        FasterForest ff1 = trainFF1(42, 32, 1);
        FasterForest ff2 = trainFF1(42, 32, 4);
        assertIdenticalForests("FF1 thread-count independence", ff1, ff2);
    }

    @Test
    public void ff1_differentSeed_producesDifferentPredictions() throws Exception {
        FasterForest ff1 = trainFF1(42, 32, 1);
        FasterForest ff2 = trainFF1(99, 32, 1);

        double[][] instances = extractInstances(dataset);
        boolean allSame = true;
        for (double[] inst : instances) {
            if (ff1.predict(inst) != ff2.predict(inst)) {
                allSame = false;
                break;
            }
        }
        assertFalse("Different seeds should produce different predictions", allSame);
    }

    // =========================================================================
    //  FasterForest2 (v2)
    // =========================================================================

    @Test
    public void ff2_sameSeed_singleThread_producesSamePredictions() throws Exception {
        FasterForest2 ff1 = trainFF2(42, 16, 1);
        FasterForest2 ff2 = trainFF2(42, 16, 1);
        assertIdenticalForests("FF2 single-thread", ff1, ff2);
    }

    @Test
    public void ff2_sameSeed_multiThread_producesSamePredictions() throws Exception {
        FasterForest2 ff1 = trainFF2(42, 16, 4);
        FasterForest2 ff2 = trainFF2(42, 16, 4);
        assertIdenticalForests("FF2 multi-thread", ff1, ff2);
    }

    @Test
    public void ff2_differentThreadCount_producesSamePredictions() throws Exception {
        FasterForest2 ff1 = trainFF2(42, 16, 1);
        FasterForest2 ff2 = trainFF2(42, 16, 4);
        assertIdenticalForests("FF2 thread-count independence", ff1, ff2);
    }

    // =========================================================================
    //  Helpers — Training
    // =========================================================================

    private FasterForest trainFF1(int seed, int numTrees, int numThreads) throws Exception {
        FasterForest ff = new FasterForest();
        ff.setNumTrees(numTrees);
        ff.setSeed(seed);
        ff.setNumFeatures(5);
        ff.setMaxDepth(0);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.setNumThreads(numThreads);
        ff.buildClassifier(dataset);
        return ff;
    }

    private FasterForest2 trainFF2(int seed, int numTrees, int numThreads) throws Exception {
        FasterForest2 ff = new FasterForest2();
        ff.setNumTrees(numTrees);
        ff.setSeed(seed);
        ff.setNumFeatures(5);
        ff.setMaxDepth(4);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.setComputeDropoutImportance(false);
        ff.setComputeInteractions(false);
        ff.setComputeInteractionsNew(false);
        ff.setNumThreads(numThreads);
        ff.buildClassifier(dataset);
        return ff;
    }

    // =========================================================================
    //  Helpers — Assertion
    // =========================================================================

    private void assertIdenticalForests(String label, FasterForest ff1, FasterForest ff2) {
        // Structural checks
        assertEquals(label + ": numTrees", ff1.getNumTrees(), ff2.getNumTrees());
        assertEquals(label + ": numAttributes", ff1.getNumAttributes(), ff2.getNumAttributes());

        // Tree-level structure comparison
        List<FasterTree> trees1 = ff1.getTrees();
        List<FasterTree> trees2 = ff2.getTrees();
        for (int t = 0; t < trees1.size(); t++) {
            assertTreesIdentical(label + " tree[" + t + "]", trees1.get(t), trees2.get(t));
        }

        // Prediction-level comparison (bit-exact)
        double[][] instances = extractInstances(dataset);
        for (int i = 0; i < instances.length; i++) {
            double p1 = ff1.predict(instances[i]);
            double p2 = ff2.predict(instances[i]);
            assertEquals(label + " prediction[" + i + "]", p1, p2, 0.0);
        }
    }

    private void assertIdenticalForests(String label, FasterForest2 ff1, FasterForest2 ff2) {
        // Structural checks
        assertEquals(label + ": numTrees", ff1.getNumTrees(), ff2.getNumTrees());
        assertEquals(label + ": numAttributes", ff1.getNumAttributes(), ff2.getNumAttributes());

        // Tree-level structure comparison
        List<FasterTree> trees1 = ff1.getTrees();
        List<FasterTree> trees2 = ff2.getTrees();
        for (int t = 0; t < trees1.size(); t++) {
            assertTreesIdentical(label + " tree[" + t + "]", trees1.get(t), trees2.get(t));
        }

        // Prediction-level comparison (bit-exact)
        double[][] instances = extractInstances(dataset);
        for (int i = 0; i < instances.length; i++) {
            double p1 = ff1.predict(instances[i]);
            double p2 = ff2.predict(instances[i]);
            assertEquals(label + " prediction[" + i + "]", p1, p2, 0.0);
        }
    }

    /**
     * Recursively compares two FasterTree structures for bit-identity.
     */
    private void assertTreesIdentical(String label, FasterTree t1, FasterTree t2) {
        assertEquals(label + ".attribute", t1.m_Attribute, t2.m_Attribute);
        assertEquals(label + ".splitPoint",
                Double.doubleToLongBits(t1.m_SplitPoint),
                Double.doubleToLongBits(t2.m_SplitPoint));

        if (t1.m_ClassProbs == null) {
            assertNull(label + ".classProbs should both be null", t2.m_ClassProbs);
        } else {
            assertNotNull(label + ".classProbs should both be non-null", t2.m_ClassProbs);
            assertEquals(label + ".classProbs.length", t1.m_ClassProbs.length, t2.m_ClassProbs.length);
            for (int i = 0; i < t1.m_ClassProbs.length; i++) {
                assertEquals(label + ".classProbs[" + i + "]",
                        Double.doubleToLongBits(t1.m_ClassProbs[i]),
                        Double.doubleToLongBits(t2.m_ClassProbs[i]));
            }
        }

        if (t1.m_Attribute != -1) { // not a leaf
            assertTreesIdentical(label + ".L", t1.sucessorLeft, t2.sucessorLeft);
            assertTreesIdentical(label + ".R", t1.sucessorRight, t2.sucessorRight);
        }
    }

    private static double[][] extractInstances(Instances data) {
        int numFeatures = data.numAttributes() - 1;
        double[][] instances = new double[data.numInstances()][numFeatures];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            int idx = 0;
            for (int a = 0; a < data.numAttributes(); a++) {
                if (a != data.classIndex()) {
                    instances[i][idx++] = inst.value(a);
                }
            }
        }
        return instances;
    }

    private static Instances loadDataset(String path) throws Exception {
        Instances data = new ConverterUtils.DataSource(path).getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
