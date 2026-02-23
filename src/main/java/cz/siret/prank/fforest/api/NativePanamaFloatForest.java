package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

/**
 * Stub for Java 17 compatibility. The real implementation lives in
 * {@code src/main/java22} and is loaded automatically on Java 22+ via
 * the multi-release JAR (META-INF/versions/22/).
 *
 * <p>On Java 17, {@link #isAvailable()} returns {@code false} and all
 * factory/prediction methods throw {@link UnsupportedOperationException}.
 */
public class NativePanamaFloatForest implements BinaryForest, Classifier, AutoCloseable {

    protected NativePanamaFloatForest() {
    }

    public static boolean isAvailable() {
        return false;
    }

    public static NativePanamaFloatForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    public static NativePanamaFloatForest fromContiguousDfsForest(ContiguousDfsForest base) {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public int getNumAttributes() {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public int getNumTrees() {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public int getMaxDepth() {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public double predict(double[] instanceAttributes) {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        throw new UnsupportedOperationException("Native float forest requires Java 22+");
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return distributionForInst(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
