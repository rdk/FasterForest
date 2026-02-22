package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

/**
 * Variant of {@link InterleavedBfsForest} using {@code long[]} node data with double-precision
 * split points, eliminating the float conversion overhead of the original.
 *
 * <p>Node layout in {@code nodeData[]} (4 longs = 32 bytes per node):
 * <pre>
 *   nodeData[node*4 + 0] = childLeft   (negative = leaf index into score[])
 *   nodeData[node*4 + 1] = childRight
 *   nodeData[node*4 + 2] = attributeIndex
 *   nodeData[node*4 + 3] = Double.doubleToRawLongBits(splitPoint)
 * </pre>
 *
 * <p>Uses BFS (breadth-first) node ordering per tree and double-precision comparisons.
 * 2 nodes fit per 64-byte cache line (vs. 4 for the float variant).
 */
public class InterleavedBfsDoubleForest implements BinaryForest, Classifier, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final int LEFT = 0;
    private static final int RIGHT = 1;
    private static final int ATTR = 2;
    private static final int SPLIT = 3;

    protected final int numTrees;
    protected final int numAttributes;

    /**
     * Interleaved node data. 4 longs per node.
     * [childLeft, childRight, attributeIndex, doubleBits(splitPoint)]
     */
    protected final long[] nodeData;

    /**
     * Leaf scores. Index 0 is unused (by design, so that -1 is not a valid leaf index).
     */
    protected final double[] score;

    protected final double invNumTrees;

//===============================================================================================//

    public InterleavedBfsDoubleForest(int numTrees, int numAttributes, long[] nodeData, double[] score) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.nodeData = nodeData;
        this.score = score;
        this.invNumTrees = 1.0 / numTrees;
    }

//===============================================================================================//
// Static factory
//===============================================================================================//

    /**
     * Build from trained FasterTrees using BFS node ordering.
     */
    public static InterleavedBfsDoubleForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        int splitNodeCount = 0;
        int leafCount = 0;
        for (FasterTree tree : trees) {
            splitNodeCount += tree.numSplitNodes();
            leafCount += tree.numLeaves();
        }

        int numTrees = trees.size();
        int m = Math.max(numTrees, splitNodeCount);

        // Temporary parallel arrays for BFS compilation
        int[] childLeft = new int[m];
        int[] childRight = new int[m];
        int[] attrIndex = new int[m];
        double[] splitPoint = new double[m];
        double[][] classProbs = new double[leafCount + 1][];

        int posSplitNodes = numTrees;
        int posClassProbs = 1; // skip 0 by design

        // Compile each tree in BFS order
        Queue<int[]> idxQueue = new ArrayDeque<>();
        Queue<FasterTree> treeQueue = new ArrayDeque<>();

        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            if (tree.isLeaf()) {
                childLeft[t] = -posClassProbs;
                childRight[t] = -posClassProbs;
                classProbs[posClassProbs] = tree.getClassProbs();
                posClassProbs++;
                continue;
            }

            idxQueue.clear();
            treeQueue.clear();
            idxQueue.add(new int[]{t});
            treeQueue.add(tree);

            while (!idxQueue.isEmpty()) {
                int nodeIdx = idxQueue.poll()[0];
                FasterTree node = treeQueue.poll();

                attrIndex[nodeIdx] = node.getAttribute();
                splitPoint[nodeIdx] = node.getSplitPoint();

                FasterTree left = node.getSucessorLeft();
                FasterTree right = node.getSucessorRight();

                if (left.isLeaf()) {
                    childLeft[nodeIdx] = -posClassProbs;
                    classProbs[posClassProbs] = left.getClassProbs();
                    posClassProbs++;
                } else {
                    int leftIdx = posSplitNodes++;
                    childLeft[nodeIdx] = leftIdx;
                    idxQueue.add(new int[]{leftIdx});
                    treeQueue.add(left);
                }

                if (right.isLeaf()) {
                    childRight[nodeIdx] = -posClassProbs;
                    classProbs[posClassProbs] = right.getClassProbs();
                    posClassProbs++;
                } else {
                    int rightIdx = posSplitNodes++;
                    childRight[nodeIdx] = rightIdx;
                    idxQueue.add(new int[]{rightIdx});
                    treeQueue.add(right);
                }
            }
        }

        // Validate
        for (int i = 1; i < classProbs.length; ++i) {
            if (classProbs[i] == null) {
                throw new RuntimeException(String.format("Found null class probs at index %d out of %d leaves", i, classProbs.length - 1));
            }
        }

        // Pack into interleaved long[] nodeData
        long[] nodeData = new long[m * 4];
        for (int i = 0; i < m; ++i) {
            int base = i << 2;
            nodeData[base]         = childLeft[i];
            nodeData[base + 1]     = childRight[i];
            nodeData[base + 2]     = attrIndex[i];
            nodeData[base + 3]     = Double.doubleToRawLongBits(splitPoint[i]);
        }

        // Compute leaf scores: p1 / (p0 + p1)
        double[] score = new double[classProbs.length];
        for (int i = 1; i < classProbs.length; ++i) {
            double p1 = classProbs[i][1];
            score[i] = p1 / (classProbs[i][0] + p1);
        }

        return new InterleavedBfsDoubleForest(numTrees, numAttributes, nodeData, score);
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
        int left = (int) nodeData[base + LEFT];
        int right = (int) nodeData[base + RIGHT];
        return Math.max(treeDepth(left), treeDepth(right)) + 1;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        final long[] nd = this.nodeData;
        final double[] sc = this.score;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = t;
            do {
                int base = node << 2;
                double sp = Double.longBitsToDouble(nd[base + SPLIT]);
                if (instanceAttributes[(int) nd[base + ATTR]] < sp) {
                    node = (int) nd[base + LEFT];
                } else {
                    node = (int) nd[base + RIGHT];
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
        final long[] nd = this.nodeData;
        final double[] sc = this.score;
        final int nt = this.numTrees;

        for (int t = 0; t < nt; ++t) {
            for (int i = 0; i < n; ++i) {
                final double[] inst = instances[i];
                int node = t;
                do {
                    int base = node << 2;
                    double sp = Double.longBitsToDouble(nd[base + SPLIT]);
                    if (inst[(int) nd[base + ATTR]] < sp) {
                        node = (int) nd[base + LEFT];
                    } else {
                        node = (int) nd[base + RIGHT];
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
