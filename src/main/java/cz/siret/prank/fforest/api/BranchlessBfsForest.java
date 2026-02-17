package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

/**
 * Branchless forest with contiguous per-tree BFS ordering.
 *
 * <p>Uses a merged {@code children[]} array (left/right interleaved at stride 2) so that
 * child selection can be done with arithmetic instead of an if/else branch:
 * <pre>
 *   node = children[node * 2 + (inst[attr] >= sp ? 1 : 0)];
 * </pre>
 * The ternary expression can compile to a {@code cmov} (conditional move) instruction,
 * avoiding branch misprediction which is the dominant cost for unpredictable tree traversal.
 *
 * <p>Each tree's nodes occupy a contiguous block with BFS ordering.
 * Separate arrays for attributeIndex and splitPoint (double precision, no conversions).
 */
public class BranchlessBfsForest implements BinaryForest, Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    protected final int numTrees;
    protected final int numAttributes;

    /**
     * Merged child array. For node i: children[i*2] = left child, children[i*2+1] = right child.
     * Negative values are leaf indices into score[].
     */
    protected final int[] children;
    protected final int[] attributeIndex;
    protected final double[] splitPoint;
    protected final double[] score;
    protected final int[] treeRoots;

    protected final double invNumTrees;

//===============================================================================================//

    public BranchlessBfsForest(int numTrees, int numAttributes,
                               int[] children, int[] attributeIndex, double[] splitPoint,
                               double[] score, int[] treeRoots) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.children = children;
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
     * Build from trained FasterTrees with contiguous per-tree BFS layout and branchless traversal.
     */
    public static BranchlessBfsForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        int numTrees = trees.size();

        int totalSplitNodes = 0;
        int totalLeaves = 0;
        int[] treeSplitCounts = new int[numTrees];
        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int sc = tree.numSplitNodes();
            treeSplitCounts[t] = Math.max(1, sc);
            totalSplitNodes += treeSplitCounts[t];
            totalLeaves += tree.numLeaves();
        }

        int[] treeRoots = new int[numTrees];
        int offset = 0;
        for (int t = 0; t < numTrees; ++t) {
            treeRoots[t] = offset;
            offset += treeSplitCounts[t];
        }
        int totalNodes = offset;

        // Merged children array (stride 2) + separate attr/split arrays
        int[] childrenArr = new int[totalNodes * 2];
        int[] attrIndex = new int[totalNodes];
        double[] splitPt = new double[totalNodes];
        double[][] classProbs = new double[totalLeaves + 1][];

        int posClassProbs = 1;

        Queue<int[]> idxQueue = new ArrayDeque<>();
        Queue<FasterTree> treeQueue = new ArrayDeque<>();

        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int root = treeRoots[t];
            int nextNode = root + 1;

            if (tree.isLeaf()) {
                childrenArr[root * 2] = -posClassProbs;
                childrenArr[root * 2 + 1] = -posClassProbs;
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
                    childrenArr[nodeIdx * 2] = -posClassProbs;
                    classProbs[posClassProbs] = left.getClassProbs();
                    posClassProbs++;
                } else {
                    int leftIdx = nextNode++;
                    childrenArr[nodeIdx * 2] = leftIdx;
                    idxQueue.add(new int[]{leftIdx});
                    treeQueue.add(left);
                }

                if (right.isLeaf()) {
                    childrenArr[nodeIdx * 2 + 1] = -posClassProbs;
                    classProbs[posClassProbs] = right.getClassProbs();
                    posClassProbs++;
                } else {
                    int rightIdx = nextNode++;
                    childrenArr[nodeIdx * 2 + 1] = rightIdx;
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

        double[] score = new double[classProbs.length];
        for (int i = 1; i < classProbs.length; ++i) {
            double p1 = classProbs[i][1];
            score[i] = p1 / (classProbs[i][0] + p1);
        }

        return new BranchlessBfsForest(numTrees, numAttributes, childrenArr, attrIndex, splitPt, score, treeRoots);
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
        int left = children[node * 2];
        int right = children[node * 2 + 1];
        return Math.max(treeDepth(left), treeDepth(right)) + 1;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        final int[] ch = this.children;
        final int[] ai = this.attributeIndex;
        final double[] sp = this.splitPoint;
        final double[] sc = this.score;
        final int[] roots = this.treeRoots;
        double sum = 0.0;

        for (int t = 0; t < numTrees; ++t) {
            int node = roots[t];
            do {
                // branchless child selection: 0 = left (attr < split), 1 = right (attr >= split)
                int sel = instanceAttributes[ai[node]] < sp[node] ? 0 : 1;
                node = ch[node * 2 + sel];
            } while (node >= 0);
            sum += sc[-node];
        }

        return sum * invNumTrees;
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        final int n = instances.length;
        final double[] sums = new double[n];
        final int[] ch = this.children;
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
                do {
                    int sel = inst[ai[node]] < sp[node] ? 0 : 1;
                    node = ch[node * 2 + sel];
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
