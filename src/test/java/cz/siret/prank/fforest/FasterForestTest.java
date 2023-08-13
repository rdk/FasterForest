package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.FlatBinaryForest;
import cz.siret.prank.fforest2.FasterForest2;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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

        ff.setNumTrees(1024);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(0);
        ff.setBagSizePercent(55);

        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);

        return ff;
    }

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

        FlatBinaryForest fbf = ff.toFlatBinaryForest();

        //assertEquals(ff.getMaxDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
    }

    @Test
    public void flattenFF2() throws Exception {
        FasterForest2 ff = setupFF2();

        ff.buildClassifier(dataset1);

        FlatBinaryForest fbf = ff.toFlatBinaryForest();

        assertEquals(ff.getMaxDepth(), fbf.getMaxDepth());
        assertEquals(ff.getNumTrees(), fbf.getNumTrees());
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