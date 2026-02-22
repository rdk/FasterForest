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
 * Separate-array forest with contiguous per-tree DFS (depth-first) node ordering.
 *
 * <p>Like {@link FlatBinaryForest} but each tree's nodes occupy a contiguous block
 * (instead of sharing roots at positions 0..numTrees-1). DFS pre-order keeps
 * parent and left-child adjacent in memory, which is ideal for sequential prefetching
 * along the most common traversal paths.
 *
 * <p>A {@code treeRoots[]} array maps tree index to root node index.
 */
public class ContiguousDfsForest implements BinaryForest, Classifier, Serializable {

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

    public ContiguousDfsForest(int numTrees, int numAttributes,
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
     * Build from trained FasterTrees with contiguous per-tree DFS layout.
     */
    public static ContiguousDfsForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
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

        // Compute contiguous start offset for each tree
        int[] treeRoots = new int[numTrees];
        int offset = 0;
        for (int t = 0; t < numTrees; ++t) {
            treeRoots[t] = offset;
            offset += treeSplitCounts[t];
        }
        int totalNodes = offset;

        int[] childLeft = new int[totalNodes];
        int[] childRight = new int[totalNodes];
        int[] attrIndex = new int[totalNodes];
        double[] splitPt = new double[totalNodes];
        double[][] classProbs = new double[totalLeaves + 1][];

        int[] posClassProbs = {1}; // mutable counter, skip 0 by design

        for (int t = 0; t < numTrees; ++t) {
            FasterTree tree = trees.get(t);
            int root = treeRoots[t];

            if (tree.isLeaf()) {
                childLeft[root] = -posClassProbs[0];
                childRight[root] = -posClassProbs[0];
                classProbs[posClassProbs[0]] = tree.getClassProbs();
                posClassProbs[0]++;
                continue;
            }

            int[] nextNode = {root + 1}; // mutable counter for this tree's block
            compileDfs(root, tree, childLeft, childRight, attrIndex, splitPt, classProbs, posClassProbs, nextNode);
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

        return new ContiguousDfsForest(numTrees, numAttributes, childLeft, childRight, attrIndex, splitPt, score, treeRoots);
    }

    /**
     * Recursively compile a split node in DFS pre-order within a contiguous block.
     */
    private static void compileDfs(int nodeIdx, FasterTree tree,
                                   int[] childLeft, int[] childRight,
                                   int[] attrIndex, double[] splitPt,
                                   double[][] classProbs, int[] posClassProbs, int[] nextNode) {
        attrIndex[nodeIdx] = tree.getAttribute();
        splitPt[nodeIdx] = tree.getSplitPoint();

        FasterTree left = tree.getSucessorLeft();
        FasterTree right = tree.getSucessorRight();

        int leftIdx = -1;
        int rightIdx = -1;

        if (left.isLeaf()) {
            childLeft[nodeIdx] = -posClassProbs[0];
            classProbs[posClassProbs[0]] = left.getClassProbs();
            posClassProbs[0]++;
        } else {
            leftIdx = nextNode[0]++;
            childLeft[nodeIdx] = leftIdx;
        }

        if (right.isLeaf()) {
            childRight[nodeIdx] = -posClassProbs[0];
            classProbs[posClassProbs[0]] = right.getClassProbs();
            posClassProbs[0]++;
        } else {
            rightIdx = nextNode[0]++;
            childRight[nodeIdx] = rightIdx;
        }

        // Recurse left first (DFS pre-order: left subtree is contiguous after parent)
        if (leftIdx >= 0) {
            compileDfs(leftIdx, left, childLeft, childRight, attrIndex, splitPt, classProbs, posClassProbs, nextNode);
        }
        if (rightIdx >= 0) {
            compileDfs(rightIdx, right, childLeft, childRight, attrIndex, splitPt, classProbs, posClassProbs, nextNode);
        }
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
