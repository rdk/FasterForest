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
 * Separate-array forest with contiguous per-tree BFS node ordering.
 *
 * <p>Combines the separate-array layout of {@link FlatBinaryForest} (which avoids
 * interleaving overhead and type conversions) with contiguous per-tree BFS node ordering
 * (which keeps frequently-accessed top-level nodes packed together).
 *
 * <p>Each tree's split nodes occupy a contiguous block in all arrays.
 * Within each block, nodes are ordered breadth-first.
 * A {@code treeRoots[]} array maps tree index to root node index.
 */
public class SeparateArraysBfsForest implements BinaryForest, Classifier, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    protected final int numTrees;
    protected final int numAttributes;
    protected final int[] childLeft;
    protected final int[] childRight;
    protected final int[] attributeIndex;
    protected final double[] splitPoint;
    protected final double[] score;
    protected final int[] treeRoots;

    protected final double invNumTrees;

//===============================================================================================//

    public SeparateArraysBfsForest(int numTrees, int numAttributes,
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
     * Build from trained FasterTrees with contiguous per-tree BFS layout and separate arrays.
     */
    public static SeparateArraysBfsForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
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

        // Separate arrays
        int[] childLeft = new int[totalNodes];
        int[] childRight = new int[totalNodes];
        int[] attrIndex = new int[totalNodes];
        double[] splitPt = new double[totalNodes];
        double[][] classProbs = new double[totalLeaves + 1][];

        int posClassProbs = 1; // skip 0 by design

        // Compile each tree in BFS order within its contiguous block
        Queue<int[]> idxQueue = new ArrayDeque<>();
        Queue<FasterTree> treeQueue = new ArrayDeque<>();

        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int root = treeRoots[t];
            int nextNode = root + 1;

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
                splitPt[nodeIdx] = node.getSplitPoint();

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

        // Compute leaf scores: p1 / (p0 + p1)
        double[] score = new double[classProbs.length];
        for (int i = 1; i < classProbs.length; ++i) {
            double p1 = classProbs[i][1];
            score[i] = p1 / (classProbs[i][0] + p1);
        }

        return new SeparateArraysBfsForest(numTrees, numAttributes, childLeft, childRight, attrIndex, splitPt, score, treeRoots);
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

        for (int t = 0; t < nt; ++t) {
            final int root = roots[t];
            for (int i = 0; i < n; ++i) {
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
