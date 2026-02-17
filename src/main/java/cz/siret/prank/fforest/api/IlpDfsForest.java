package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Contiguous DFS forest with instruction-level parallelism (ILP) batch prediction.
 *
 * <p>Processes 4 instances simultaneously through the same tree. The 4 traversals
 * are interleaved in the same loop body, allowing the CPU's out-of-order engine to
 * overlap memory loads across lanes: while lane 0 waits for a cache line, lanes 1-3
 * can issue their own loads and comparisons.
 *
 * <p>Same data layout as {@link ContiguousDfsForest} — separate arrays, contiguous
 * per-tree DFS ordering. Only the batch prediction loop is different.
 */
public class IlpDfsForest implements BinaryForest, Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    protected final int numTrees;
    protected final int numAttributes;
    protected final int[] childLeft;
    protected final int[] childRight;
    protected final int[] attributeIndex;
    protected final double[] splitPoint;
    protected final double[] score;
    protected final int[] treeRoots;

    protected transient final double invNumTrees;

//===============================================================================================//

    public IlpDfsForest(int numTrees, int numAttributes,
                        int[] childLeft, int[] childRight,
                        int[] attributeIndex, double[] splitPoint,
                        double[] score, int[] treeRoots) {
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
     * Build from trained FasterTrees (delegates to ContiguousDfsForest layout).
     */
    public static IlpDfsForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        // Reuse ContiguousDfsForest's construction, then wrap with our prediction logic
        ContiguousDfsForest base = ContiguousDfsForest.fromFasterTrees(numAttributes, trees);
        return new IlpDfsForest(
                base.numTrees, base.numAttributes,
                base.childLeft, base.childRight,
                base.attributeIndex, base.splitPoint,
                base.score, base.treeRoots
        );
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
        final double[] sp = this.splitPoint;
        final double[] sc = this.score;
        final int[] roots = this.treeRoots;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = roots[t];
            while (true) {
                int attr = ai[node];
                if (instanceAttributes[attr] < sp[node]) {
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
        final double[] sp = this.splitPoint;
        final double[] sc = this.score;
        final int[] roots = this.treeRoots;
        final int nt = this.numTrees;

        final int n4 = n - (n & 3); // round down to multiple of 4

        for (int t = 0; t < nt; ++t) {
            final int root = roots[t];

            // ILP: process 4 instances at a time through the same tree
            for (int i = 0; i < n4; i += 4) {
                final double[] i0 = instances[i];
                final double[] i1 = instances[i + 1];
                final double[] i2 = instances[i + 2];
                final double[] i3 = instances[i + 3];

                int n0 = root, n1 = root, n2 = root, n3 = root;

                // Interleaved traversal: 4 independent paths through the same tree.
                // CPU out-of-order engine overlaps loads across lanes.
                while ((n0 >= 0) | (n1 >= 0) | (n2 >= 0) | (n3 >= 0)) {
                    if (n0 >= 0) {
                        if (i0[ai[n0]] < sp[n0]) n0 = cl[n0]; else n0 = cr[n0];
                    }
                    if (n1 >= 0) {
                        if (i1[ai[n1]] < sp[n1]) n1 = cl[n1]; else n1 = cr[n1];
                    }
                    if (n2 >= 0) {
                        if (i2[ai[n2]] < sp[n2]) n2 = cl[n2]; else n2 = cr[n2];
                    }
                    if (n3 >= 0) {
                        if (i3[ai[n3]] < sp[n3]) n3 = cl[n3]; else n3 = cr[n3];
                    }
                }

                sums[i]     += sc[-n0];
                sums[i + 1] += sc[-n1];
                sums[i + 2] += sc[-n2];
                sums[i + 3] += sc[-n3];
            }

            // Scalar tail for remaining instances
            for (int i = n4; i < n; ++i) {
                final double[] inst = instances[i];
                int node = root;
                while (true) {
                    int attr = ai[node];
                    if (inst[attr] < sp[node]) {
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

        double inv = this.invNumTrees;
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
