package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.*;
import cz.siret.prank.fforest2.FasterForest2;
import cz.siret.prank.ffutils.ATimer;
import cz.siret.prank.ffutils.StrUtils;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;

import static cz.siret.prank.fforest.api.OptimizingFlatBinaryForest.NodeOrderings.*;
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

        ff.setNumTrees(128);
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
            double[] classProbs_ff = ff.distributionForInst(inst);
            double[] classProbs_fbf = fbf.distributionForInst(inst);
            // here they will not be equal
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
            double[] classProbs_ff = ff.distributionForInst(inst);
            double[] classProbs_fbf = fbf.distributionForInst(inst);

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
    public void optimizingFlatForestLegacy() throws Exception {
        FasterForest ff = setupFF();
        ff.setNumTrees(512);
        ff.setMaxDepth(0);

        ff.buildClassifier(dataset1);
        LegacyFlatBinaryForest fbf = ff.toFlatBinaryForest();

        OptimizingFlatBinaryForest optimizingForest = new OptimizingFlatBinaryForest(fbf);

        // rebuild without training
        LegacyFlatBinaryForest optimizedForest = optimizingForest.buildOptimizedForest();
        //System.out.println("Tree positions: " + optimizingForest.getTreePositions());
        //System.out.println("Score positions: " + optimizingForest.getScorePositions());
        //System.out.println(StrUtils.toStr(fbf));
        //System.out.println(StrUtils.toStr(optimizedForest));

        testEqualStructure(fbf, optimizedForest);
        testEqualPredictions(dataset1, fbf, optimizedForest, 0.000000000000001d);

        // train counts
        for (Instance inst : dataset1) {
            optimizingForest.predict(inst.toDoubleArray());
        }

        // rebuild with training
        LegacyFlatBinaryForest optimizedForest2 = optimizingForest.buildOptimizedForest();
        //System.out.println("Tree positions: " + optimizingForest.getTreePositions());
        //System.out.println("Score positions: " + optimizingForest.getScorePositions());
        //System.out.println("ORIG: " + StrUtils.toStr(fbf));
        //System.out.println("OPTI: " + StrUtils.toStr(optimizedForest2));

        testEqualStructure(fbf, optimizedForest2);
        testEqualPredictions(dataset1, fbf, optimizedForest2, 0.000000000000001d);
    }

    @Test
    public void optimizingFlatForestLegacyBenchmark() throws Exception {
        FasterForest ff = setupFF();
        ff.setNumTrees(100);
        ff.setMaxDepth(0);

        ff.buildClassifier(dataset1);
        LegacyFlatBinaryForest fbf = ff.toFlatBinaryForest();

        OptimizingFlatBinaryForest optimizingForest = new OptimizingFlatBinaryForest(fbf);
        // train counts
        for (Instance inst : dataset1) {
            optimizingForest.predict(inst.toDoubleArray());
        }
        LegacyFlatBinaryForest optimizedForest = optimizingForest.buildOptimizedForest();
        //System.out.println("Tree positions: " + optimizingForest.getTreePositions());
        //System.out.println("Score positions: " + optimizingForest.getScorePositions());
        //System.out.println(StrUtils.toStr(fbf));
        //System.out.println(StrUtils.toStr(optimizedForest));

        testEqualStructure(fbf, optimizedForest);
        testEqualPredictions(dataset1, fbf, optimizedForest, 0.000000000000001d);

        ShortLegacyFlatBinaryForest shortForest = ShortLegacyFlatBinaryForest.from(fbf);

        testEqualStructure(fbf, shortForest);
        testEqualPredictions(dataset1, fbf, shortForest, 0.0000001d);

        int n = 200;
        int m = 5;
        for (int i=0; i!=m; ++i) {
            System.out.printf("Original: %d ms\n", benchPredictions(n, dataset1, ff));
            System.out.printf("Flat: %d ms\n", benchPredictions(n, dataset1, fbf));
            System.out.printf("Optimized: %d ms\n", benchPredictions(n, dataset1, optimizedForest));
            System.out.printf("Short: %d ms\n", benchPredictions(n, dataset1, shortForest));
            System.out.println("------");
        }
    }

    @Test
    public void optimizingFlatForestLegacyBenchmark2() throws Exception {
        FasterForest ff = setupFF();
        ff.setNumTrees(100);
        ff.setMaxDepth(0);

        ff.buildClassifier(dataset1);
        LegacyFlatBinaryForest fbf = ff.toFlatBinaryForest();

        OptimizingFlatBinaryForest optimizingForest = new OptimizingFlatBinaryForest(fbf);
        // train counts
        for (Instance inst : dataset1) {
            optimizingForest.predict(inst.toDoubleArray());
        }
        LegacyFlatBinaryForest of_DEFAULT = optimizingForest.buildOptimizedForest();
        LegacyFlatBinaryForest of_COUNT = optimizingForest.buildOptimizedForest(BY_COUNT, BY_COUNT);
        LegacyFlatBinaryForest of_TREE_COUNT = optimizingForest.buildOptimizedForest(BY_TREE_COUNT, BY_TREE_COUNT);
        LegacyFlatBinaryForest of_TREE_DEPTH_COUNT = optimizingForest.buildOptimizedForest(BY_TREE_DEPTH_COUNT, BY_TREE_DEPTH_COUNT);
        LegacyFlatBinaryForest of_TREE_DEPTH = optimizingForest.buildOptimizedForest(BY_TREE_DEPTH, BY_TREE_DEPTH);
        LegacyFlatBinaryForest of_DEPTH = optimizingForest.buildOptimizedForest(BY_DEPTH, BY_DEPTH);
        LegacyFlatBinaryForest of_DEPTH_COUNT = optimizingForest.buildOptimizedForest(BY_DEPTH_COUNT, BY_DEPTH_COUNT);
        FlatBinaryForest fbf_NO_LAGACY = ff.toFlatBinaryForest(false);

        ShortLegacyFlatBinaryForest shortForest = ShortLegacyFlatBinaryForest.from(fbf);

        int n = 100;
        int m = 5;
        for (int i=0; i!=m; ++i) {
            System.out.printf("Original: %d ms\n", benchPredictions(n, dataset1, ff));
            System.out.printf("Flat: %d ms\n", benchPredictions(n, dataset1, fbf));
            System.out.printf("Flat no legacy: %d ms\n", benchPredictions(n, dataset1, fbf_NO_LAGACY));
            System.out.printf("OPT_DEFAULT : %d ms\n", benchPredictions(n, dataset1, of_DEFAULT));
            System.out.printf("OPT_COUNT  : %d ms\n", benchPredictions(n, dataset1, of_COUNT));
            System.out.printf("OPT_TREE_COUNT : %d ms\n", benchPredictions(n, dataset1, of_TREE_COUNT));
            System.out.printf("OPT_TREE_DEPTH_COUNT : %d ms\n", benchPredictions(n, dataset1, of_TREE_DEPTH_COUNT));
            System.out.printf("OPT_TREE_DEPTH : %d ms\n", benchPredictions(n, dataset1, of_TREE_DEPTH));
            System.out.printf("OPT_DEPTH : %d ms\n", benchPredictions(n, dataset1, of_DEPTH));
            System.out.printf("OPT_DEPTH_COUNT : %d ms\n", benchPredictions(n, dataset1, of_DEPTH_COUNT));
            System.out.printf("Short: %d ms\n", benchPredictions(n, dataset1, shortForest));
            System.out.println("------");
        }
    }

    private long benchPredictions(int n, Instances dataset, BinaryForest forest) {
        int m = dataset.size();
        double[][] instances = new double[m][];
        for (int i=0; i!=m; ++i) {
            instances[i] = dataset.get(i).toDoubleArray();
        }

        ATimer timer = ATimer.startTimer();
        for (int i=0; i!=n; ++i) {
            for (double[] inst : instances) {
                forest.predict(inst);
            }
        }
        return timer.getTime();
    }

    private void testEqualStructure(BinaryForest forestA, BinaryForest forestB) {
        assertEquals(forestA.getNumTrees(),      forestB.getNumTrees());
        assertEquals(forestA.getNumAttributes(), forestB.getNumAttributes());
        assertEquals(forestA.getMaxDepth(),      forestB.getMaxDepth());
    }


    private void testEqualPredictions(Instances dataset, BinaryForest forestA, BinaryForest forestB, double DELTA) {
        for (Instance inst : dataset) {
            double[] classProbs_fbf = forestA.distributionForInst(inst);
            double[] classProbs_opt = forestB.distributionForInst(inst);

            assertArrayEquals(classProbs_opt, classProbs_fbf, DELTA);
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
            double[] classProbs_fbf = fbf.distributionForInst(inst);
            // here they will not be equal
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
            double[] classProbs_fbf = fbf.distributionForInst(inst);

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