package cz.siret.prank.fforest.api;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 * Cache-optimized FlatBinaryForest with:
 * <ul>
 *   <li>P1: Interleaved node data in a single int[] array (stride 4) with float-encoded split points</li>
 *   <li>P2: BFS (breadth-first) node ordering per tree for optimal cache locality</li>
 *   <li>P3: Inlined tree traversal with local variable caching</li>
 * </ul>
 *
 * Node layout in {@code nodeData[]} (4 ints per node):
 * <pre>
 *   nodeData[node*4 + 0] = childLeft   (negative = leaf index into score[])
 *   nodeData[node*4 + 1] = childRight
 *   nodeData[node*4 + 2] = attributeIndex
 *   nodeData[node*4 + 3] = Float.floatToRawIntBits(splitPoint)
 * </pre>
 *
 * Indices 0..numTrees-1 are root nodes of each tree.
 * Leaf pointers use negative values: score[-childLeft] or score[-childRight].
 *
 * @see FlatBinaryForestBuilder#buildInterleavedBfsForest(int, java.util.List)
 */
public class InterleavedBfsForest implements BinaryForest, Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    private static final int LEFT = 0;
    private static final int RIGHT = 1;
    private static final int ATTR = 2;
    private static final int SPLIT = 3;
    private static final int STRIDE = 4;

    protected final int numTrees;
    protected final int numAttributes;

    /**
     * Interleaved node data. 4 ints per node.
     * [childLeft, childRight, attributeIndex, floatBits(splitPoint)]
     */
    protected final int[] nodeData;

    /**
     * Leaf scores. Index 0 is unused (by design, so that -1 is not a valid leaf index).
     */
    protected final float[] score;

    protected transient final double invNumTrees;

//===============================================================================================//

    public InterleavedBfsForest(int numTrees, int numAttributes, int[] nodeData, float[] score) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.nodeData = nodeData;
        this.score = score;
        this.invNumTrees = 1.0 / numTrees;
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
            max = Math.max(max, treeDepth(t));
        }
        return max;
    }

    private int treeDepth(int node) {
        if (node < 0) {
            return 1;
        }
        int base = node << 2;
        int left = nodeData[base + LEFT];
        int right = nodeData[base + RIGHT];
        return Math.max(treeDepth(left), treeDepth(right)) + 1;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        final int[] nd = this.nodeData;
        final float[] sc = this.score;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = t;
            do {
                int base = node << 2;
                float sp = Float.intBitsToFloat(nd[base + SPLIT]);
                if ((float) instanceAttributes[nd[base + ATTR]] < sp) {
                    node = nd[base + LEFT];
                } else {
                    node = nd[base + RIGHT];
                }
            } while (node >= 0);
            sum += sc[-node];
        }

        return sum * invNumTrees;
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        final int n = instances.length;
        final double[] sums = new double[n];
        final int[] nd = this.nodeData;
        final float[] sc = this.score;
        final int nt = this.numTrees;

        for (int t = 0; t < nt; ++t) {
            for (int i = 0; i < n; ++i) {
                final double[] inst = instances[i];
                int node = t;
                do {
                    int base = node << 2;
                    float sp = Float.intBitsToFloat(nd[base + SPLIT]);
                    if ((float) inst[nd[base + ATTR]] < sp) {
                        node = nd[base];         // childLeft
                    } else {
                        node = nd[base + 1];     // childRight
                    }
                } while (node >= 0);
                sums[i] += sc[-node];
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
