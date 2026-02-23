package cz.siret.prank.fforest;

import cz.siret.prank.fforest.api.*;
import cz.siret.prank.fforest2.FasterForest2;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Prediction-equivalence tests for all BinaryForest implementations.
 *
 * Verifies that all inference forest representations produce equivalent predictions
 * when derived from the same trained FasterForest model. Two prediction families are
 * tested independently:
 * <ul>
 *   <li>Score-based: pre-normalized leaf scores averaged across trees</li>
 *   <li>Legacy class-probs: raw class probabilities summed then normalized</li>
 * </ul>
 */
public class BinaryForestInferenceTest {

    static final String DATA_DIR = "src/test/resources/data/";

    /** Bit-identical tolerance for double-precision forests with power-of-2 tree count */
    static final double DELTA_EXACT = 0.0;
    /** Tolerance for float-precision forests compared against each other */
    static final double DELTA_FLOAT = 1e-6;
    /** Tolerance for legacy ground truth (FasterForest vs LegacyFlat, ULP-level differences) */
    static final double DELTA_LEGACY = 1e-15;

    static Instances dataset;
    static double[][] instances;

    // Trained FasterForest (legacy ground-truth anchor)
    static FasterForest fasterForest;

    // Score-based reference (double precision)
    static BinaryForest flatBinaryForest;

    // Score-based double-precision forests
    static BinaryForest interleavedBfsDoubleForest;
    static BinaryForest contiguousBfsDoubleForest;
    static BinaryForest separateArraysBfsForest;
    static BinaryForest branchlessBfsForest;
    static BinaryForest contiguousDfsForest;
    static BinaryForest ilpDfsForest;
    static BinaryForest blockedIlpDfsForest;

    // Score-based float-precision forests
    static BinaryForest interleavedBfsForest;
    static BinaryForest flatBinaryFloatForest;
    static BinaryForest ilpDfsFloatForest;
    static BinaryForest blockedIlpDfsFloatForest;

    // Legacy class-probs reference
    static BinaryForest legacyFlatBinaryForest;

    // Legacy class-probs forests
    static BinaryForest shortLegacyForest;
    static BinaryForest superShortLegacyForest;

    // =========================================================================
    //  Setup
    // =========================================================================

    @BeforeClass
    public static void setUp() throws Exception {
        dataset = loadDataset(DATA_DIR + "p2rank-train.arff.gz");
        instances = instancesToArrays(dataset);

        fasterForest = new FasterForest();
        fasterForest.setNumTrees(128);
        fasterForest.setSeed(42);
        fasterForest.setNumFeatures(5);
        fasterForest.setMaxDepth(0);
        fasterForest.setBagSizePercent(55);
        fasterForest.setCalcOutOfBag(false);
        fasterForest.setComputeImportances(false);
        fasterForest.buildClassifier(dataset);

        // Score-based reference
        flatBinaryForest = convert(FasterForestConverter.ForestType.FlatBinaryForest);

        // Score-based double-precision
        interleavedBfsDoubleForest = convert(FasterForestConverter.ForestType.InterleavedBfsDoubleForest);
        contiguousBfsDoubleForest  = convert(FasterForestConverter.ForestType.ContiguousBfsDoubleForest);
        separateArraysBfsForest    = convert(FasterForestConverter.ForestType.SeparateArraysBfsForest);
        branchlessBfsForest        = convert(FasterForestConverter.ForestType.BranchlessBfsForest);
        contiguousDfsForest        = convert(FasterForestConverter.ForestType.ContiguousDfsForest);
        ilpDfsForest               = convert(FasterForestConverter.ForestType.IlpDfsForest);
        blockedIlpDfsForest        = convert(FasterForestConverter.ForestType.BlockedIlpDfsForest);

        // Score-based float-precision
        interleavedBfsForest       = convert(FasterForestConverter.ForestType.InterleavedBfsForest);
        flatBinaryFloatForest      = convert(FasterForestConverter.ForestType.FlatBinaryFloatForest);
        ilpDfsFloatForest          = convert(FasterForestConverter.ForestType.IlpDfsFloatForest);
        blockedIlpDfsFloatForest   = convert(FasterForestConverter.ForestType.BlockedIlpDfsFloatForest);

        // Legacy class-probs
        legacyFlatBinaryForest = convert(FasterForestConverter.ForestType.LegacyFlatBinaryForest);
        shortLegacyForest      = convert(FasterForestConverter.ForestType.ShortFlatBinaryForest);
        superShortLegacyForest = convert(FasterForestConverter.ForestType.SuperShortLegacyFlatBinaryForest);
    }

    // =========================================================================
    //  Test 1: Score-based double-precision forests match reference
    // =========================================================================

    @Test
    public void allDoubleScoreForests_matchReference() {
        Map<String, BinaryForest> forests = new LinkedHashMap<>();
        forests.put("InterleavedBfsDouble", interleavedBfsDoubleForest);
        forests.put("ContiguousBfsDouble", contiguousBfsDoubleForest);
        forests.put("SeparateArraysBfs", separateArraysBfsForest);
        forests.put("BranchlessBfs", branchlessBfsForest);
        forests.put("ContiguousDfs", contiguousDfsForest);
        forests.put("IlpDfs", ilpDfsForest);
        forests.put("BlockedIlpDfs", blockedIlpDfsForest);

        for (Map.Entry<String, BinaryForest> entry : forests.entrySet()) {
            assertStructureMatches(entry.getKey(), flatBinaryForest, entry.getValue());
            assertPredictionsMatch(entry.getKey(), flatBinaryForest, entry.getValue(), DELTA_EXACT);
        }
    }

    // =========================================================================
    //  Test 2: Score-based float-precision forests match reference
    // =========================================================================

    @Test
    public void allFloatScoreForests_matchReference() {
        // Float forests use float-precision split points, which can change tree decisions
        // (path divergence, not just accumulation precision). Compare them against each other
        // using FlatBinaryFloatForest as the float-family reference.
        Map<String, BinaryForest> forests = new LinkedHashMap<>();
        forests.put("InterleavedBfs", interleavedBfsForest);
        forests.put("IlpDfsFloat", ilpDfsFloatForest);
        forests.put("BlockedIlpDfsFloat", blockedIlpDfsFloatForest);

        for (Map.Entry<String, BinaryForest> entry : forests.entrySet()) {
            assertStructureMatches(entry.getKey(), flatBinaryFloatForest, entry.getValue());
            assertPredictionsMatch(entry.getKey(), flatBinaryFloatForest, entry.getValue(), DELTA_FLOAT);
        }
    }

    // =========================================================================
    //  Test 3: Legacy class-probs forests match reference
    // =========================================================================

    @Test
    public void allLegacyFloatForests_matchReference() {
        Map<String, BinaryForest> forests = new LinkedHashMap<>();
        forests.put("ShortLegacy", shortLegacyForest);
        forests.put("SuperShortLegacy", superShortLegacyForest);

        for (Map.Entry<String, BinaryForest> entry : forests.entrySet()) {
            assertStructureMatches(entry.getKey(), legacyFlatBinaryForest, entry.getValue());
            assertPredictionsMatch(entry.getKey(), legacyFlatBinaryForest, entry.getValue(), DELTA_FLOAT);
        }
    }

    // =========================================================================
    //  Test 4: Legacy ground truth (anchor to original FasterForest)
    // =========================================================================

    @Test
    public void legacyFlat_matchesFasterForest() throws Exception {
        for (int i = 0; i < dataset.size(); i++) {
            Instance inst = dataset.get(i);
            double[] ffProbs = fasterForest.distributionForInstance(inst);
            double[] legacyProbs = legacyFlatBinaryForest.distributionForInst(inst);
            assertArrayEquals(
                    "Legacy vs FasterForest mismatch at instance " + i,
                    ffProbs, legacyProbs, DELTA_LEGACY);
        }
    }

    // =========================================================================
    //  Test 5: Native forests
    // =========================================================================

    @Test
    public void nativeDoubleForests_matchReference() {
        Assume.assumeTrue("Native library not available", NativePanamaForest.isAvailable());

        try (NativePanamaForest nativeForest = (NativePanamaForest) convert(
                FasterForestConverter.ForestType.NativePanamaForest)) {
            assertCoreStructureMatches("NativePanama", flatBinaryForest, nativeForest);
            assertPredictionsMatch("NativePanama", flatBinaryForest, nativeForest, DELTA_EXACT);
        }

        if (NativePanamaForestAvx2.isAvx2Available()) {
            try (NativePanamaForest nativeAvx2 = (NativePanamaForest) convert(
                    FasterForestConverter.ForestType.NativePanamaForestAvx2)) {
                assertCoreStructureMatches("NativePanamaAvx2", flatBinaryForest, nativeAvx2);
                assertPredictionsMatch("NativePanamaAvx2", flatBinaryForest, nativeAvx2, DELTA_EXACT);
            }
        }
    }

    @Test
    public void nativeFloatForests_matchReference() {
        Assume.assumeTrue("Native library not available", NativePanamaFloatForest.isAvailable());

        // Native float forests share path divergence with Java float forests,
        // so compare against the float-family reference (FlatBinaryFloatForest).
        try (NativePanamaFloatForest nativeFloat = (NativePanamaFloatForest) convert(
                FasterForestConverter.ForestType.NativePanamaFloatForest)) {
            assertCoreStructureMatches("NativePanamaFloat", flatBinaryFloatForest, nativeFloat);
            assertPredictionsMatch("NativePanamaFloat", flatBinaryFloatForest, nativeFloat, DELTA_FLOAT);
        }

        if (NativePanamaFloatForestAvx2.isAvx2Available()) {
            try (NativePanamaFloatForest nativeFloatAvx2 = (NativePanamaFloatForest) convert(
                    FasterForestConverter.ForestType.NativePanamaFloatForestAvx2)) {
                assertCoreStructureMatches("NativePanamaFloatAvx2", flatBinaryFloatForest, nativeFloatAvx2);
                assertPredictionsMatch("NativePanamaFloatAvx2", flatBinaryFloatForest, nativeFloatAvx2, DELTA_FLOAT);
            }
        }
    }

    @Test
    public void nativePanama_bothConstructionPaths() {
        Assume.assumeTrue("Native library not available", NativePanamaForest.isAvailable());

        try (NativePanamaForest fromTrees = NativePanamaForest.fromFasterTrees(
                     fasterForest.getFeatureVectorLength(), fasterForest.getTrees());
             NativePanamaForest fromDfs = NativePanamaForest.fromContiguousDfsForest(
                     (ContiguousDfsForest) contiguousDfsForest)) {

            for (int i = 0; i < instances.length; i++) {
                double fromTreesResult = fromTrees.predict(instances[i]);
                double fromDfsResult = fromDfs.predict(instances[i]);
                assertEquals("NativePanama construction path mismatch at instance " + i,
                        fromTreesResult, fromDfsResult, DELTA_EXACT);
            }
        }
    }

    // =========================================================================
    //  Test 6: Batch vs single predictions
    // =========================================================================

    @Test
    public void batchMatchesSingle_ilpDfs() {
        assertBatchMatchesSingle("IlpDfs", ilpDfsForest, DELTA_EXACT);
    }

    @Test
    public void batchMatchesSingle_blockedIlpDfs() {
        assertBatchMatchesSingle("BlockedIlpDfs", blockedIlpDfsForest, DELTA_EXACT);
    }

    @Test
    public void batchMatchesSingle_ilpDfsFloat() {
        assertBatchMatchesSingle("IlpDfsFloat", ilpDfsFloatForest, DELTA_EXACT);
    }

    @Test
    public void batchMatchesSingle_blockedIlpDfsFloat() {
        assertBatchMatchesSingle("BlockedIlpDfsFloat", blockedIlpDfsFloatForest, DELTA_EXACT);
    }

    @Test
    public void batchMatchesSingle_legacy() {
        assertBatchMatchesSingle("LegacyFlat", legacyFlatBinaryForest, DELTA_EXACT);
    }

    @Test
    public void batchMatchesSingle_nativePanama() {
        Assume.assumeTrue("Native library not available", NativePanamaForest.isAvailable());

        try (NativePanamaForest nativeForest = (NativePanamaForest) convert(
                FasterForestConverter.ForestType.NativePanamaForest)) {
            assertBatchMatchesSingle("NativePanama", nativeForest, DELTA_EXACT);
        }
    }

    // =========================================================================
    //  Test 7: ILP edge cases (batch sizes that exercise remainder paths)
    // =========================================================================

    @Test
    public void ilpForests_batchEdgeCases() {
        int[] batchSizes = {1, 2, 3, 4, 5, 7};

        for (int batchSize : batchSizes) {
            double[][] batch = Arrays.copyOf(instances, batchSize);

            assertBatchMatchesSingleForBatch("IlpDfs batchSize=" + batchSize,
                    ilpDfsForest, batch, DELTA_EXACT);

            assertBatchMatchesSingleForBatch("IlpDfsFloat batchSize=" + batchSize,
                    ilpDfsFloatForest, batch, DELTA_EXACT);
        }
    }

    // =========================================================================
    //  Test 8: FasterForest2 conversions
    // =========================================================================

    @Test
    public void ff2_conversionsProduceConsistentPredictions() throws Exception {
        FasterForest2 ff2 = new FasterForest2();
        ff2.setNumTrees(8);
        ff2.setSeed(42);
        ff2.setNumFeatures(5);
        ff2.setMaxDepth(4);
        ff2.setBagSizePercent(55);
        ff2.setCalcOutOfBag(false);
        ff2.setComputeImportances(false);
        ff2.setComputeDropoutImportance(false);
        ff2.setComputeInteractions(false);
        ff2.setComputeInteractionsNew(false);
        ff2.buildClassifier(dataset);

        BinaryForest ff2Flat = FasterForestConverter.convertFasterForest(ff2,
                FasterForestConverter.ForestType.FlatBinaryForest);
        BinaryForest ff2Dfs = FasterForestConverter.convertFasterForest(ff2,
                FasterForestConverter.ForestType.ContiguousDfsForest);
        BinaryForest ff2Interleaved = FasterForestConverter.convertFasterForest(ff2,
                FasterForestConverter.ForestType.InterleavedBfsDoubleForest);
        BinaryForest ff2Float = FasterForestConverter.convertFasterForest(ff2,
                FasterForestConverter.ForestType.FlatBinaryFloatForest);
        BinaryForest ff2FloatIlp = FasterForestConverter.convertFasterForest(ff2,
                FasterForestConverter.ForestType.IlpDfsFloatForest);

        // Double forests should match exactly
        assertPredictionsMatch("FF2 ContiguousDfs", ff2Flat, ff2Dfs, DELTA_EXACT);
        assertPredictionsMatch("FF2 InterleavedBfsDouble", ff2Flat, ff2Interleaved, DELTA_EXACT);

        // Float forests compared against each other (path divergence vs double)
        assertPredictionsMatch("FF2 FlatBinaryFloat vs IlpDfsFloat", ff2Float, ff2FloatIlp, DELTA_FLOAT);
    }

    // =========================================================================
    //  Helpers
    // =========================================================================

    private static BinaryForest convert(FasterForestConverter.ForestType type) {
        return FasterForestConverter.convertFasterForest(fasterForest, type);
    }

    private static Instances loadDataset(String path) throws Exception {
        Instances data = new ConverterUtils.DataSource(path).getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private static double[][] instancesToArrays(Instances data) {
        double[][] result = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            result[i] = data.get(i).toDoubleArray();
        }
        return result;
    }

    private static void assertStructureMatches(String name, BinaryForest ref, BinaryForest candidate) {
        assertEquals(name + " numTrees mismatch", ref.getNumTrees(), candidate.getNumTrees());
        assertEquals(name + " numAttributes mismatch", ref.getNumAttributes(), candidate.getNumAttributes());
        assertEquals(name + " maxDepth mismatch", ref.getMaxDepth(), candidate.getMaxDepth());
    }

    /** Like assertStructureMatches but skips maxDepth (native forests may not track it). */
    private static void assertCoreStructureMatches(String name, BinaryForest ref, BinaryForest candidate) {
        assertEquals(name + " numTrees mismatch", ref.getNumTrees(), candidate.getNumTrees());
        assertEquals(name + " numAttributes mismatch", ref.getNumAttributes(), candidate.getNumAttributes());
    }

    private static void assertPredictionsMatch(String name, BinaryForest ref, BinaryForest candidate, double delta) {
        for (int i = 0; i < instances.length; i++) {
            double refResult = ref.predict(instances[i]);
            double candidateResult = candidate.predict(instances[i]);
            assertEquals(name + " prediction mismatch at instance " + i,
                    refResult, candidateResult, delta);
        }
    }

    private static void assertBatchMatchesSingle(String name, BinaryForest forest, double delta) {
        assertBatchMatchesSingleForBatch(name, forest, instances, delta);
    }

    private static void assertBatchMatchesSingleForBatch(String name, BinaryForest forest,
                                                          double[][] batch, double delta) {
        double[] batchResults = forest.predictForBatch(batch);
        assertEquals(name + " batch result length mismatch", batch.length, batchResults.length);
        for (int i = 0; i < batch.length; i++) {
            double singleResult = forest.predict(batch[i]);
            assertEquals(name + " batch vs single mismatch at instance " + i,
                    singleResult, batchResults[i], delta);
        }
    }

}
