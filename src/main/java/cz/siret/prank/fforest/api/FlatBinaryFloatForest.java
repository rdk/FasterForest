package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Float-precision variant of {@link FlatBinaryForest}.
 *
 * <p>Uses {@code float[]} for split points and leaf scores, halving their memory
 * footprint compared to {@code double[]}. More nodes fit per cache line, reducing
 * cache misses during tree traversal.
 *
 * <p>Instance attributes are cast to float before comparison to match the stored precision.
 *
 * @see FlatBinaryForest
 */
public class FlatBinaryFloatForest implements BinaryForest, Classifier, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    protected final int numTrees;
    protected final int numAttributes;
    protected final int[] childRight;
    protected final int[] childLeft;
    protected final int[] attributeIndex;
    protected final float[] splitPoint;
    protected final float[] score;

    protected final double numTreesAsDouble;
    protected transient int maxDepth = -1;
    protected transient int[] treeDepths;

//===============================================================================================//

    public FlatBinaryFloatForest(int numTrees, int numAttributes, int[] childLeft, int[] childRight,
                                 int[] attributeIndex, float[] splitPoint, float[] score) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.childLeft = childLeft;
        this.childRight = childRight;
        this.attributeIndex = attributeIndex;
        this.splitPoint = splitPoint;
        this.score = score;
        this.numTreesAsDouble = numTrees;
    }

//===============================================================================================//
// Static factories
//===============================================================================================//

    /**
     * Build from trained FasterTrees (builds double forest via FlatBinaryForestBuilder, converts to float).
     */
    public static FlatBinaryFloatForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        FlatBinaryForest base = FlatBinaryForestBuilder.buildFromFasterTrees(numAttributes, trees, false);
        return fromFlatBinaryForest(base);
    }

    /**
     * Convert an existing FlatBinaryForest to float precision.
     */
    public static FlatBinaryFloatForest fromFlatBinaryForest(FlatBinaryForest base) {
        return new FlatBinaryFloatForest(
                base.numTrees, base.numAttributes,
                base.childLeft, base.childRight,
                base.attributeIndex,
                toFloatArray(base.splitPoint),
                toFloatArray(base.score)
        );
    }

    private static float[] toFloatArray(double[] src) {
        float[] dst = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            dst[i] = (float) src[i];
        }
        return dst;
    }

//===============================================================================================//

    @Override
    public int getNumAttributes() {
        return numAttributes;
    }

    @Override
    public int getNumTrees() {
        return numTrees;
    }

    @Override
    public int getMaxDepth() {
        if (maxDepth < 0) {
            calculateDepths();
        }
        return maxDepth;
    }

    public int[] getTreeDepths() {
        if (treeDepths == null) {
            calculateDepths();
        }
        return treeDepths;
    }

//===============================================================================================//

    private void calculateDepths() {
        treeDepths = calculateTreeDepths();
        maxDepth = Arrays.stream(treeDepths).max().getAsInt();
    }

    private int[] calculateTreeDepths() {
        int[] depths = new int[numTrees];
        for (int i = 0; i != numTrees; ++i) {
            depths[i] = calculateTreeDepth(i);
        }
        return depths;
    }

    private int calculateTreeDepth(int tree) {
        if (tree < 0) {
            return 1;
        }
        int left = calculateTreeDepth(childLeft[tree]);
        int right = calculateTreeDepth(childRight[tree]);
        return Math.max(left, right) + 1;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        double sum = 0d;

        for (int i = 0; i != numTrees; ++i) {
            sum += predictTree(i, instanceAttributes);
        }

        return sum / numTreesAsDouble;
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        int n = instances.length;
        double[] sums = new double[n];

        for (int t = 0; t != numTrees; ++t) {
            for (int i = 0; i != n; ++i) {
                sums[i] += predictTree(t, instances[i]);
            }
        }

        for (int i = 0; i != n; ++i) {
            sums[i] /= numTrees;
        }
        return sums;
    }

//===============================================================================================//

    protected double predictTree(int tree, double[] instanceAttributes) {
        int currentNode = tree;

        while (true) {
            int attr = attributeIndex[currentNode];

            if ((float) instanceAttributes[attr] < splitPoint[currentNode]) {
                currentNode = childLeft[currentNode];
            } else {
                currentNode = childRight[currentNode];
            }

            if (currentNode < 0) {
                return score[-currentNode];
            }
        }
    }

    public double[] evalTrees(double[] instanceAttributes) {
        double[] res = new double[numTrees];
        for (int i = 0; i != numTrees; ++i) {
            res[i] = predictTree(i, instanceAttributes);
        }
        return res;
    }

//===============================================================================================//

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // do nothing
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
