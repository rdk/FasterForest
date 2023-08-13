package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.FlatBinaryForest;
import cz.siret.prank.fforest2.FasterForest2;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.Assert.*;

/**
 *
 */
public class FasterForestTest {

    static String dataDir = "src/test/resources/data/";

    Instances dataset1;


    private static Instances loadDataset(String path) throws Exception {
        Instances data = new ConverterUtils.DataSource(path).getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    @Before
    public void init() throws Exception {
        dataset1 = loadDataset(dataDir + "p2rank-train.arff.gz");
    }

//===============================================================================================//

    private FasterForest setupFF() {
        FasterForest ff = new FasterForest();

        ff.setNumTrees(64);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(0);
        ff.setBagSizePercent(55);

        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);

        return ff;
    }


//===============================================================================================//

    @Test
    public void createFF() {
        FasterForest ff = setupFF();

        assertTrue(ff != null);

        ff.getTechnicalInformation();
        ff.getCapabilities();

    }

    @Test
    public void trainFF() throws Exception {
        FasterForest ff = setupFF();

        ff.buildClassifier(dataset1);

    }

    @Test
    public void featureImportancesFF() throws Exception {
        FasterForest ff = setupFF();

        ff.setComputeImportances(true);

        ff.buildClassifier(dataset1);
        ff.getFeatureImportances();
        ff.toString();
    }


    @Test
    public void flattenFF() throws Exception {
        FasterForest ff = setupFF();
        ff.buildClassifier(dataset1);

        FlatBinaryForest fbf = ff.toFlatBinaryForest(false);

        assertEquals(ff.calculateMaxTreeDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
        assertEquals(ff.getFeatureVectorLength(), fbf.getNumAttributes());

        for (Instance inst : dataset1) {
            double[] classProbs_ff = ff.distributionForInstance(inst);
            double[] classProbs_fbf = fbf.distributionForInstance(inst);
            assertArrayEquals(classProbs_ff, classProbs_fbf, 0.000000000000001d);
        }

    }

    @Test
    public void flattenFFLegacy() throws Exception {
        FasterForest ff = setupFF();

        ff.buildClassifier(dataset1);

        FlatBinaryForest fbf = ff.toFlatBinaryForest();

        assertEquals(ff.calculateMaxTreeDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
        assertEquals(ff.getFeatureVectorLength(), fbf.getNumAttributes());

        System.out.println("Orig tree depths:" + Arrays.toString(ff.calculateTreeDepths()));
        System.out.println("Flat tree depths:" + Arrays.toString(fbf.getTreeDepths()));

        for (Instance inst : dataset1) {
            double[] classProbs_ff = ff.distributionForInstance(inst);
            double[] classProbs_fbf = fbf.distributionForInstance(inst);

            assertArrayEquals(classProbs_ff, classProbs_fbf, 0.000000000000001d);

            //double[][] eval_ff = ff.evalTrees(inst.toDoubleArray());
            //double[] eval_fbf = fbf.evalTrees(inst.toDoubleArray());
            //for (int i=0; i!=ff.getNumTrees(); ++i) {
            //    System.out.printf(" Tree %d: FF:%s FBF:%f%n", i, Arrays.toString(eval_ff[i]), eval_fbf[i]);
            //}
            //System.out.println("Class probs FF:" + Arrays.toString(classProbs_ff) + ", FBF:" + Arrays.toString(classProbs_fbf));
        }

    }

    @Test
    public void flattenFF2() throws Exception {
        FasterForest2 ff = setupFF2();

        ff.buildClassifier(dataset1);

        FlatBinaryForest fbf = ff.toFlatBinaryForest(false);

        assertEquals(ff.calculateMaxTreeDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
        assertEquals(ff.getFeatureVectorLength(), fbf.getNumAttributes());

        for (Instance inst : dataset1) {
            double[] classProbs_ff = ff.distributionForInstance(inst);
            double[] classProbs_fbf = fbf.distributionForInstance(inst);

            assertArrayEquals(classProbs_ff, classProbs_fbf, 0.000000000000001d);
        }
        
    }

    @Test
    public void flattenFF2Legacy() throws Exception {
        FasterForest2 ff = setupFF2();

        ff.buildClassifier(dataset1);

        FlatBinaryForest fbf = ff.toFlatBinaryForest();

        assertEquals(ff.calculateMaxTreeDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
        assertEquals(ff.getFeatureVectorLength(), fbf.getNumAttributes());

        for (Instance inst : dataset1) {
            double[] classProbs_ff = ff.distributionForInstance(inst);
            double[] classProbs_fbf = fbf.distributionForInstance(inst);

            assertArrayEquals(classProbs_ff, classProbs_fbf, 0.000000000000001d);
        }

    }

//===============================================================================================//

    private FasterForest2 setupFF2() {
        FasterForest2 ff = new FasterForest2();

        ff.setNumTrees(8);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(4);
        ff.setBagSizePercent(55);

        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.setComputeDropoutImportance(false);
        ff.setComputeInteractions(false);
        ff.setComputeInteractionsNew(false);

        return ff;
    }

//===============================================================================================//

    @Test
    public void createFF2() {
        FasterForest2 ff = setupFF2();

        assertTrue(ff != null);

        ff.getTechnicalInformation();
        ff.getCapabilities();

    }

    @Test
    public void trainFF2() throws Exception {
        FasterForest2 ff = setupFF2();

        ff.buildClassifier(dataset1);
        ff.toString();
    }

    @Test
    public void featureImportancesFF2() throws Exception {
        FasterForest2 ff = setupFF2();

        ff.setComputeImportances(true);

        ff.buildClassifier(dataset1);
        ff.getFeatureImportances();
        ff.toString();
    }

// TODO fix failing test featureImportancesNewFF2

//    @Test
//    public void featureImportancesNewFF2() throws Exception {
//        FasterForest2 ff = setupFF2();
//
//        ff.setBagSizePercent(100);
//        ff.setCalcOutOfBag(true);
//        ff.setComputeDropoutImportance(true);
//
//        ff.buildClassifier(dataset1);
//        ff.getFeatureImportances();
//        ff.toString();
//    }


}