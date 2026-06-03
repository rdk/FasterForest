package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterForest;
import org.junit.Assume;
import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Guards against the native (Panama FFM) path silently rotting. The other native tests
 * {@link Assume}-skip when the library is unavailable, so CI can be green while never running
 * native code. CI runs this with {@code -Dci.native.required=true} on a platform that ships a
 * native binary (linux-x86_64), turning "native unavailable" into a hard failure; elsewhere it
 * skips gracefully. When available it also smoke-checks that native predictions match the Java
 * reference (bit-exact for the double scalar path).
 */
public class NativeAvailabilityTest {

    @Test
    public void nativePanama_loadsAndMatchesJava() throws Exception {
        boolean available = NativePanamaForest.isAvailable();

        if (Boolean.getBoolean("ci.native.required")) {
            assertTrue("CI requires the native library, but NativePanamaForest.isAvailable() == false "
                    + "(the committed native binary failed to load on this platform)", available);
        }
        Assume.assumeTrue("native library not available on this platform", available);

        System.out.println("Native SIMD level: " + NativePanamaForest.simdLevel());

        Instances data = new ConverterUtils.DataSource("src/test/resources/data/p2rank-train.arff.gz").getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

        FasterForest ff = new FasterForest();
        ff.setNumTrees(20);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(0);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.buildClassifier(data);

        BinaryForest java = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.FlatBinaryForest);
        BinaryForest nat = FasterForestConverter.convertFasterForest(ff, FasterForestConverter.ForestType.NativePanamaForest);

        int n = Math.min(100, data.size());
        for (int i = 0; i < n; i++) {
            double[] x = data.get(i).toDoubleArray();
            assertEquals("native vs java prediction mismatch at instance " + i, java.predict(x), nat.predict(x), 1e-9);
        }
    }
}
