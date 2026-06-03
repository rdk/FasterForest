package cz.siret.prank.fforest;

import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Pinned golden baseline for the trained-model prediction path. Trains a small, fixed
 * {@link FasterForest} (the faithful reference) and pins a handful of prediction aggregates to
 * {@code prediction-golden.csv}. Fast (~50 trees) and runs in the normal suite.
 *
 * <p>Unlike the cross-representation equivalence tests (which would all drift together under an
 * algorithm change), this anchors absolute output, so it catches drift in the training or
 * aggregation code path. Regenerate after an intentional change:
 * <pre>./gradlew test --tests '*PredictionGoldenTest' -Dgolden.regenerate=true</pre>
 */
public class PredictionGoldenTest {

    static final String DATA = "src/test/resources/data/p2rank-train.arff.gz";
    static final Path GOLDEN = Path.of("src/test/resources/prediction-golden.csv");
    static final int NUM_TREES = 50;
    static final int SAMPLE = 5; // first N per-instance predictions pinned

    private double[] referencePredictions() throws Exception {
        Instances data = new ConverterUtils.DataSource(DATA).getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

        FasterForest ff = new FasterForest();
        ff.setNumTrees(NUM_TREES);
        ff.setSeed(42);
        ff.setNumFeatures(5);
        ff.setMaxDepth(0);
        ff.setBagSizePercent(55);
        ff.setCalcOutOfBag(false);
        ff.setComputeImportances(false);
        ff.buildClassifier(data);

        double[][] arr = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) arr[i] = data.get(i).toDoubleArray();
        return ff.predictForBatch(arr); // faithful sum-then-normalize
    }

    private Map<String, Double> aggregates(double[] p) {
        double sum = 0, min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : p) { sum += v; if (v < min) min = v; if (v > max) max = v; }
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("count", (double) p.length);
        m.put("sum", sum);
        m.put("mean", sum / p.length);
        m.put("min", min);
        m.put("max", max);
        for (int i = 0; i < SAMPLE; i++) m.put("p" + i, p[i]);
        return m;
    }

    @Test
    public void predictions_matchGolden() throws Exception {
        Map<String, Double> actual = aggregates(referencePredictions());

        if (Boolean.getBoolean("golden.regenerate")) {
            List<String> lines = new ArrayList<>();
            actual.forEach((k, v) -> lines.add(k + "," + Double.toString(v)));
            Files.write(GOLDEN, lines);
            System.out.println("Regenerated " + GOLDEN + " (" + lines.size() + " values)");
            return;
        }

        assertTrue("Golden file missing; regenerate with -Dgolden.regenerate=true", Files.exists(GOLDEN));
        Map<String, Double> golden = new LinkedHashMap<>();
        for (String line : Files.readAllLines(GOLDEN)) {
            if (line.isBlank()) continue;
            int c = line.indexOf(',');
            golden.put(line.substring(0, c), Double.parseDouble(line.substring(c + 1)));
        }

        for (Map.Entry<String, Double> e : golden.entrySet()) {
            double exp = e.getValue();
            Double act = actual.get(e.getKey());
            assertTrue("Missing aggregate: " + e.getKey(), act != null);
            // absolute for small [0,1] predictions; relative cushion for large sums. ULP-tolerant,
            // but an algorithm change moves values far more than this.
            double tol = Math.max(1e-9, Math.abs(exp) * 1e-10);
            assertEquals("Golden mismatch for " + e.getKey(), exp, act, tol);
        }
        assertEquals("Golden/actual key count differs", golden.size(), actual.size());
    }
}
