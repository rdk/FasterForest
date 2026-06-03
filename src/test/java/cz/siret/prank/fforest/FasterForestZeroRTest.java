package cz.siret.prank.fforest;

import cz.siret.prank.fforest2.FasterForest2;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Degenerate-model (ZeroR) path: when training data has only the class attribute, the forest falls
 * back to a ZeroR model. The attribute-array prediction paths (predict / distributionForAttributes /
 * predictForBatch) must serve ZeroR's constant prior instead of NPE-ing on a null bagger (TODO #7).
 */
public class FasterForestZeroRTest {

    /** Dataset with only a nominal class attribute: 3x class "0", 1x class "1". ZeroR applies
     *  Laplace smoothing → counts [3,1]+1 = [4,2]/6 → prior [2/3, 1/3]. */
    private Instances onlyClassData() {
        ArrayList<Attribute> atts = new ArrayList<>();
        ArrayList<String> classVals = new ArrayList<>(Arrays.asList("0", "1"));
        atts.add(new Attribute("@@class@@", classVals));
        Instances data = new Instances("degenerate", atts, 0);
        data.setClassIndex(0);
        for (double cls : new double[]{0, 0, 0, 1}) {
            data.add(new DenseInstance(1.0, new double[]{cls}));
        }
        return data;
    }

    @Test
    public void fasterForest_zeroRFallback_doesNotThrow() throws Exception {
        FasterForest ff = new FasterForest();
        ff.setNumTrees(10);
        ff.setSeed(42);
        ff.buildClassifier(onlyClassData());

        assertArrayEquals("ZeroR prior", new double[]{2.0 / 3, 1.0 / 3},
                ff.distributionForAttributes(new double[]{0}, 2), 1e-9);
        assertEquals("predict = positive prior", 1.0 / 3, ff.predict(new double[]{0}), 1e-9);
        assertArrayEquals("batch = constant prior", new double[]{1.0 / 3, 1.0 / 3, 1.0 / 3},
                ff.predictForBatch(new double[][]{{0}, {0}, {0}}), 1e-9);

        // auxiliary methods must reflect the degenerate single-tree model, not the configured count
        assertEquals("ZeroR model has one (leaf) tree", 1, ff.getNumTrees());
        ff.calculateMaxTreeDepth();                              // must not AIOOBE (loops getNumTrees())
        assertEquals(1, ff.evalTrees(new double[]{0}).length);
    }

    @Test
    public void fasterForest2_zeroRFallback_doesNotThrow() throws Exception {
        FasterForest2 ff = new FasterForest2();
        ff.setNumTrees(10);
        ff.setSeed(42);
        ff.buildClassifier(onlyClassData());

        assertArrayEquals(new double[]{2.0 / 3, 1.0 / 3}, ff.distributionForAttributes(new double[]{0}, 2), 1e-9);
        assertEquals(1.0 / 3, ff.predict(new double[]{0}), 1e-9);
        assertArrayEquals(new double[]{1.0 / 3, 1.0 / 3}, ff.predictForBatch(new double[][]{{0}, {0}}), 1e-9);

        assertEquals("ZeroR model has one (leaf) tree", 1, ff.getNumTrees());
        ff.calculateMaxTreeDepth();
        assertEquals(1, ff.evalTrees(new double[]{0}).length);
    }
}
