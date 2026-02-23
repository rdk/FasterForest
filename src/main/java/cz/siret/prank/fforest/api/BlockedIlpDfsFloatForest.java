package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

/**
 * Contiguous DFS forest with blocked instruction-level parallelism (ILP) batch prediction
 * and float-precision split points and scores.
 *
 * <p>Identical strategy to {@link BlockedIlpDfsForest} but uses float-precision arrays
 * for metadata, further improving cache density.
 */
public class BlockedIlpDfsFloatForest implements BinaryForest, Classifier, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final int BLOCK_SIZE = 512;

    protected final int numTrees;
    protected final int numAttributes;
    protected final int[] childLeft;
    protected final int[] childRight;
    protected final int[] attributeIndex;
    protected final float[] splitPoint;
    protected final float[] score;
    protected final int[] treeRoots;

    protected final double invNumTrees;

//===============================================================================================//

    public BlockedIlpDfsFloatForest(int numTrees, int numAttributes,
                                    int[] childLeft, int[] childRight,
                                    int[] attributeIndex, float[] splitPoint,
                                    float[] score, int[] treeRoots) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.childLeft = childLeft;
        this.childRight = childRight;
        this.attributeIndex = attributeIndex;
        this.splitPoint = splitPoint;
        this.score = score;
        this.treeRoots = treeRoots;
        this.invNumTrees = 1.0 / numTrees;
    }

//===============================================================================================//
// Static factory
//===============================================================================================//

    /**
     * Build from trained FasterTrees (delegates to ContiguousDfsForest layout, converts to float).
     */
    public static BlockedIlpDfsFloatForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        ContiguousDfsForest base = ContiguousDfsForest.fromFasterTrees(numAttributes, trees);
        return new BlockedIlpDfsFloatForest(
                base.numTrees, base.numAttributes,
                base.childLeft, base.childRight,
                base.attributeIndex,
                toFloatArray(base.splitPoint),
                toFloatArray(base.score),
                base.treeRoots
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
        int max = 0;
        for (int t = 0; t < numTrees; ++t) {
            max = Math.max(max, treeDepth(treeRoots[t]));
        }
        return max;
    }

    private int treeDepth(int node) {
        if (node < 0) {
            return 1;
        }
        return Math.max(treeDepth(childLeft[node]), treeDepth(childRight[node])) + 1;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        final int[] cl = this.childLeft;
        final int[] cr = this.childRight;
        final int[] ai = this.attributeIndex;
        final float[] sp = this.splitPoint;
        final float[] sc = this.score;
        final int[] roots = this.treeRoots;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = roots[t];
            while (true) {
                int attr = ai[node];
                if ((float) instanceAttributes[attr] < sp[node]) {
                    node = cl[node];
                } else {
                    node = cr[node];
                }
                if (node < 0) {
                    sum += sc[-node];
                    break;
                }
            }
        }

        return sum * invNumTrees;
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        final int n = instances.length;
        final double[] sums = new double[n];
        final int[] cl = this.childLeft;
        final int[] cr = this.childRight;
        final int[] ai = this.attributeIndex;
        final float[] sp = this.splitPoint;
        final float[] sc = this.score;
        final int[] roots = this.treeRoots;
        final int nt = this.numTrees;

        for (int iStart = 0; iStart < n; iStart += BLOCK_SIZE) {
            final int iEnd = Math.min(iStart + BLOCK_SIZE, n);
            final int iEnd4 = iStart + ((iEnd - iStart) & ~3);

            for (int t = 0; t < nt; ++t) {
                final int root = roots[t];

                for (int i = iStart; i < iEnd4; i += 4) {
                    final double[] i0 = instances[i];
                    final double[] i1 = instances[i + 1];
                    final double[] i2 = instances[i + 2];
                    final double[] i3 = instances[i + 3];

                    int n0 = root, n1 = root, n2 = root, n3 = root;

                    while ((n0 >= 0) | (n1 >= 0) | (n2 >= 0) | (n3 >= 0)) {
                        if (n0 >= 0) {
                            if ((float) i0[ai[n0]] < sp[n0]) n0 = cl[n0]; else n0 = cr[n0];
                        }
                        if (n1 >= 0) {
                            if ((float) i1[ai[n1]] < sp[n1]) n1 = cl[n1]; else n1 = cr[n1];
                        }
                        if (n2 >= 0) {
                            if ((float) i2[ai[n2]] < sp[n2]) n2 = cl[n2]; else n2 = cr[n2];
                        }
                        if (n3 >= 0) {
                            if ((float) i3[ai[n3]] < sp[n3]) n3 = cl[n3]; else n3 = cr[n3];
                        }
                    }

                    sums[i]     += sc[-n0];
                    sums[i + 1] += sc[-n1];
                    sums[i + 2] += sc[-n2];
                    sums[i + 3] += sc[-n3];
                }

                for (int i = iEnd4; i < iEnd; ++i) {
                    final double[] inst = instances[i];
                    int node = root;
                    while (true) {
                        int attr = ai[node];
                        if ((float) inst[attr] < sp[node]) {
                            node = cl[node];
                        } else {
                            node = cr[node];
                        }
                        if (node < 0) {
                            sums[i] += sc[-node];
                            break;
                        }
                    }
                }
            }
        }

        final double inv = this.invNumTrees;
        for (int i = 0; i < n; ++i) {
            sums[i] *= inv;
        }
        return sums;
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
