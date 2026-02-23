package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.*;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests conversion of Weka RandomForest to FasterTreeForest.
 */
public class WekaRandomForestConverterTest {

    static final String DATA_DIR = "src/test/resources/data/";

    static Instances dataset;
    static RandomForest wekaForest;
    static FasterTreeForest convertedForest;

    @BeforeClass
    public static void setUp() throws Exception {
        dataset = loadDataset(DATA_DIR + "p2rank-train.arff.gz");

        wekaForest = new RandomForest();
        wekaForest.setNumIterations(128);
        wekaForest.setSeed(42);
        wekaForest.setNumFeatures(5);
        wekaForest.setMaxDepth(0);
        wekaForest.setBagSizePercent(55);
        wekaForest.setCalcOutOfBag(false);
        wekaForest.setNumExecutionSlots(1);
        wekaForest.buildClassifier(dataset);

        convertedForest = WekaRandomForestConverter.toFasterTreeForest(wekaForest);
    }

    @Test
    public void convertedForest_hasCorrectStructure() {
        assertEquals(dataset.numAttributes(), convertedForest.getNumAttributes());

        List<FasterTree> trees = convertedForest.getTrees();
        assertEquals(128, trees.size());

        for (FasterTree tree : trees) {
            assertNotNull(tree);
            assertFalse("Root should not be a leaf", tree.isLeaf());
        }
    }

    @Test
    public void convertedForest_predictionsMatchWeka() throws Exception {
        // Aggregate predictions from converted FasterTrees the same way FasterForest does:
        // sum per-tree leaf probs, then normalize.
        int numClasses = dataset.numClasses();

        for (int i = 0; i < dataset.size(); i++) {
            Instance inst = dataset.get(i);
            double[] wekaProbs = wekaForest.distributionForInstance(inst);

            double[] instanceAttrs = inst.toDoubleArray();
            double[] sums = new double[numClasses];
            for (FasterTree tree : convertedForest.getTrees()) {
                double[] treeProbs = tree.distributionForAttributes(instanceAttrs);
                for (int c = 0; c < numClasses; c++) {
                    sums[c] += treeProbs[c];
                }
            }
            Utils.normalize(sums);

            assertArrayEquals(
                "Prediction mismatch at instance " + i,
                wekaProbs, sums, 1e-10
            );
        }
    }

    @Test
    public void convertedForest_worksWithFasterForestConverter() {
        BinaryForest flat = FasterForestConverter.convertFasterForest(
            convertedForest, FasterForestConverter.ForestType.FlatBinaryForest);

        assertNotNull(flat);
        assertEquals(128, flat.getNumTrees());
        assertEquals(dataset.numAttributes(), flat.getNumAttributes());

        // Verify predictions are valid probabilities
        for (int i = 0; i < dataset.size(); i++) {
            double p = flat.predict(dataset.get(i).toDoubleArray());
            assertTrue("Probability out of range at instance " + i, p >= 0.0 && p <= 1.0);
        }
    }

    private static Instances loadDataset(String path) throws Exception {
        Instances data = new ConverterUtils.DataSource(path).getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

}
