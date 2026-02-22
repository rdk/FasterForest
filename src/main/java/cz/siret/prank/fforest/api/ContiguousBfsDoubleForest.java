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
 * Double-precision interleaved BFS forest with contiguous per-tree memory layout.
 *
 * <p>Unlike {@link InterleavedBfsDoubleForest} where tree roots share positions 0..numTrees-1
 * (fragmenting each tree's memory), this variant allocates each tree's nodes in a contiguous
 * block. When traversing a tree, all accessed nodes are in adjacent memory, maximizing
 * cache locality and prefetcher effectiveness.
 *
 * <p>Node layout in {@code nodeData[]} (4 longs = 32 bytes per node):
 * <pre>
 *   nodeData[node*4 + 0] = childLeft   (negative = leaf index into score[])
 *   nodeData[node*4 + 1] = childRight
 *   nodeData[node*4 + 2] = attributeIndex
 *   nodeData[node*4 + 3] = Double.doubleToRawLongBits(splitPoint)
 * </pre>
 *
 * <p>Memory layout:
 * <pre>
 *   [tree0: root, level1, level2, ...][tree1: root, level1, level2, ...]...
 * </pre>
 *
 * <p>A {@code treeRoots[]} array maps tree index to root node index.
 */
public class ContiguousBfsDoubleForest implements BinaryForest, Classifier, Serializable {

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

    /**
     * Root node index for each tree. {@code treeRoots[t]} is the node index
     * of tree t's root in {@code nodeData}.
     */
    protected final int[] treeRoots;

    protected final double invNumTrees;

//===============================================================================================//

    public ContiguousBfsDoubleForest(int numTrees, int numAttributes, long[] nodeData, double[] score, int[] treeRoots) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.nodeData = nodeData;
        this.score = score;
        this.treeRoots = treeRoots;
        this.invNumTrees = 1.0 / numTrees;
    }

//===============================================================================================//
// Static factory
//===============================================================================================//

    /**
     * Build from trained FasterTrees with contiguous per-tree BFS layout.
     */
    public static ContiguousBfsDoubleForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        int numTrees = trees.size();

        // Count split nodes and leaves per tree
        int totalSplitNodes = 0;
        int totalLeaves = 0;
        int[] treeSplitCounts = new int[numTrees];
        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int sc = tree.numSplitNodes();
            treeSplitCounts[t] = Math.max(1, sc); // at least 1 slot for leaf-only trees
            totalSplitNodes += treeSplitCounts[t];
            totalLeaves += tree.numLeaves();
        }

        // Compute contiguous start offset for each tree
        int[] treeRoots = new int[numTrees];
        int offset = 0;
        for (int t = 0; t < numTrees; ++t) {
            treeRoots[t] = offset;
            offset += treeSplitCounts[t];
        }
        int totalNodes = offset;

        // Temporary parallel arrays
        int[] childLeft = new int[totalNodes];
        int[] childRight = new int[totalNodes];
        int[] attrIndex = new int[totalNodes];
        double[] splitPoint = new double[totalNodes];
        double[][] classProbs = new double[totalLeaves + 1][];

        int posClassProbs = 1; // skip 0 by design

        // Compile each tree in BFS order within its contiguous block
        Queue<int[]> idxQueue = new ArrayDeque<>();
        Queue<FasterTree> treeQueue = new ArrayDeque<>();

        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int root = treeRoots[t];
            int nextNode = root + 1; // next available slot within this tree's block

            if (tree.isLeaf()) {
                childLeft[root] = -posClassProbs;
                childRight[root] = -posClassProbs;
                classProbs[posClassProbs] = tree.getClassProbs();
                posClassProbs++;
                continue;
            }

            idxQueue.clear();
            treeQueue.clear();
            idxQueue.add(new int[]{root});
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
                    int leftIdx = nextNode++;
                    childLeft[nodeIdx] = leftIdx;
                    idxQueue.add(new int[]{leftIdx});
                    treeQueue.add(left);
                }

                if (right.isLeaf()) {
                    childRight[nodeIdx] = -posClassProbs;
                    classProbs[posClassProbs] = right.getClassProbs();
                    posClassProbs++;
                } else {
                    int rightIdx = nextNode++;
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
        long[] nodeData = new long[totalNodes * 4];
        for (int i = 0; i < totalNodes; ++i) {
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

        return new ContiguousBfsDoubleForest(numTrees, numAttributes, nodeData, score, treeRoots);
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
        final int[] roots = this.treeRoots;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = roots[t];
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
        final int[] roots = this.treeRoots;
        final int nt = this.numTrees;

        for (int t = 0; t < nt; ++t) {
            final int root = roots[t];
            for (int i = 0; i < n; ++i) {
                final double[] inst = instances[i];
                int node = root;
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
