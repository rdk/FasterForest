package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

/**
 *
 */
public class FlatBinaryForestBuilder {

    /**
     * Use only positive class probability
     */
    boolean useOnlyPositiveClassProb = false;

//===============================================================================================//

    int[] childRight;
    int[] childLeft;
    int[] attributeIndex;
    double[] splitPoint;

    double[][] classProbs;

    int posSplitNodes = 0;
    int posClassProbs = 1;  // starting at 1 and skipping 0, to be able to use negative values in childLeft/Right as index in classProbs (*-1)

//===============================================================================================//

//    /**
//     * @param trees
//     * @param useOnlyPositiveClassProbability Use only positive class probability p_class[1] instead of ratio p_class[1] / (p_class[0] + p_class[1])
//     */
//    public FlatBinaryForest buildFromFasterTrees(int numAttributes, List<FasterTree> trees, boolean useOnlyPositiveClassProbability) {
//        useOnlyPositive = useOnlyPositiveClassProbability;
//        return buildFromFasterTrees(numAttributes, trees);
//    }


    /**
     *
     * @param numAttributes input vector lenght
     * @param trees
     * @return
     */
    public FlatBinaryForest buildFromFasterTrees(int numAttributes, List<FasterTree> trees) {
        return buildFromFasterTrees(numAttributes, trees, false);
    }

    /**
     *
     * @param numAttributes input vector lenght
     * @param trees
     * @return
     */
    public LegacyFlatBinaryForest buildFromFasterTreesLegacy(int numAttributes, List<FasterTree> trees) {
        return (LegacyFlatBinaryForest)buildFromFasterTrees(numAttributes, trees, true);
    }


    /**
     *
     * @param numAttributes input vector lenght
     * @param trees
     * @return
     */
    public FlatBinaryForest buildFromFasterTrees(int numAttributes, List<FasterTree> trees, boolean legacyClassProbs) {

        int splitNodes = 0;
        int leaves = 0;

        for (FasterTree tree : trees) {
            splitNodes += tree.numSplitNodes();
            leaves += tree.numLeaves();
        }

        int numTrees = trees.size();
        int m = Math.max(numTrees, splitNodes); // at least one node for tree

        childRight = new int[m];
        childLeft = new int[m];
        attributeIndex = new int[m];
        splitPoint = new double[m];
        classProbs = new double[leaves+1][];

        posSplitNodes = numTrees; // leave first n as root nodes for each tree

        for (int i=0; i!=numTrees; ++i) {
            compileTree(i, trees.get(i));
        }

        // check nulls
        int nulls = 0;
        for (int i=1; i!=classProbs.length; ++i) {   // at i=0 classProbs is null by design
            if (classProbs[i] == null) {
                nulls++;
            }
        }
        if (nulls > 0) {
            throw new RuntimeException(String.format("Found %d null class probs out of %d leaves", nulls, classProbs.length-1));
        }

        if (legacyClassProbs) {
            return new LegacyFlatBinaryForest(trees.size(), numAttributes, childLeft, childRight, attributeIndex, splitPoint, classProbs);
        } else {
            double[] scores = calculateScoresFromProbs(classProbs);
            return new FlatBinaryForest(trees.size(), numAttributes, childLeft, childRight, attributeIndex, splitPoint, scores);
        }
        
    }

    private double[] calculateScoresFromProbs(double[][] classProbs) {
        int n = classProbs.length;
        double[] scores = new double[n];
        for (int i=1; i!=classProbs.length; ++i) {   // at i=0 classProbs is null by design
            if (classProbs[i] == null) {
                throw new RuntimeException(String.format("classProbs[%d] of %d is null", i, n));
            }
            scores[i] = getScoreFromProbs(classProbs[i]);
        }
        return scores;
    }

    private void compileTree(int treeIdx, FasterTree tree) {
        if (tree.isLeaf()) {
            childLeft[treeIdx] = -posClassProbs;
            childRight[treeIdx] = -posClassProbs;
            classProbs[posClassProbs] = tree.getClassProbs();
            posClassProbs++;
        } else {
            compileSplitNode(treeIdx, tree);
        }
    }

    private void compileSplitNode(int treeIdx, FasterTree tree) {
        attributeIndex[treeIdx] = tree.getAttribute();
        splitPoint[treeIdx] = tree.getSplitPoint();

        FasterTree left = tree.getSucessorLeft();
        FasterTree right = tree.getSucessorRight();

        int leftIdx = -1;
        int rightIdx = -1;

        if (left.isLeaf()) {
            childLeft[treeIdx] = -posClassProbs;
            classProbs[posClassProbs] = left.getClassProbs();
            posClassProbs++;
        } else {
            leftIdx = posSplitNodes++;
            childLeft[treeIdx] = leftIdx;
        }

        if (right.isLeaf()) {
            childRight[treeIdx] = -posClassProbs;
            classProbs[posClassProbs] = right.getClassProbs();
            posClassProbs++;
        } else {
            rightIdx = posSplitNodes++;
            childRight[treeIdx] = rightIdx;
        }

        if (leftIdx >= 0) {
            compileSplitNode(leftIdx, tree.getSucessorLeft());
        }
        if (rightIdx >= 0) {
            compileSplitNode(rightIdx, tree.getSucessorRight());
        }
    }

    private double getScoreFromProbs(double[] classProbs) {
        double p1 = classProbs[1];

        if (useOnlyPositiveClassProb) {
            return p1;
        } else {
            return p1 / (classProbs[0] + p1);
        }
    }

//===============================================================================================//
// InterleavedBfsForest builder
//===============================================================================================//

    // Temporary parallel arrays used during BFS compilation, then packed into interleaved format
    private int[] bfsChildLeft;
    private int[] bfsChildRight;
    private int[] bfsAttributeIndex;
    private double[] bfsSplitPoint;
    private double[][] bfsClassProbs;
    private int bfsPosSplitNodes;
    private int bfsPosClassProbs;

    /**
     * Build an InterleavedBfsForest from FasterTrees.
     * Uses BFS node ordering and interleaved int[] node data layout.
     *
     * @param numAttributes input vector length
     * @param trees list of trained FasterTree instances
     * @return optimized InterleavedBfsForest
     */
    public InterleavedBfsForest buildInterleavedBfsForest(int numAttributes, List<FasterTree> trees) {
        int splitNodes = 0;
        int leaves = 0;

        for (FasterTree tree : trees) {
            splitNodes += tree.numSplitNodes();
            leaves += tree.numLeaves();
        }

        int numTrees = trees.size();
        int m = Math.max(numTrees, splitNodes);

        bfsChildLeft = new int[m];
        bfsChildRight = new int[m];
        bfsAttributeIndex = new int[m];
        bfsSplitPoint = new double[m];
        bfsClassProbs = new double[leaves + 1][];

        bfsPosSplitNodes = numTrees;
        bfsPosClassProbs = 1;  // skip 0 by design

        for (int i = 0; i < numTrees; ++i) {
            compileTreeBfs(i, trees.get(i));
        }

        // validate
        for (int i = 1; i < bfsClassProbs.length; ++i) {
            if (bfsClassProbs[i] == null) {
                throw new RuntimeException(String.format("Found null class probs at index %d out of %d leaves", i, bfsClassProbs.length - 1));
            }
        }

        // Pack into interleaved int[] nodeData and float[] score
        int[] nodeData = new int[m * 4];
        for (int i = 0; i < m; ++i) {
            int base = i << 2;
            nodeData[base]     = bfsChildLeft[i];
            nodeData[base + 1] = bfsChildRight[i];
            nodeData[base + 2] = bfsAttributeIndex[i];
            nodeData[base + 3] = Float.floatToRawIntBits((float) bfsSplitPoint[i]);
        }

        float[] score = new float[bfsClassProbs.length];
        for (int i = 1; i < bfsClassProbs.length; ++i) {
            score[i] = (float) getScoreFromProbs(bfsClassProbs[i]);
        }

        // Release temporary arrays
        bfsChildLeft = null;
        bfsChildRight = null;
        bfsAttributeIndex = null;
        bfsSplitPoint = null;
        bfsClassProbs = null;

        return new InterleavedBfsForest(numTrees, numAttributes, nodeData, score);
    }

    /**
     * Compile a single tree using BFS ordering.
     * Nodes are allocated breadth-first so that top-level nodes are contiguous in memory.
     */
    private void compileTreeBfs(int treeIdx, FasterTree tree) {
        if (tree.isLeaf()) {
            // Leaf-only tree: both children point to same leaf
            bfsChildLeft[treeIdx] = -bfsPosClassProbs;
            bfsChildRight[treeIdx] = -bfsPosClassProbs;
            bfsClassProbs[bfsPosClassProbs] = tree.getClassProbs();
            bfsPosClassProbs++;
            return;
        }

        // BFS queue: pairs of (allocated node index, FasterTree node)
        Queue<int[]> idxQueue = new ArrayDeque<>();
        Queue<FasterTree> treeQueue = new ArrayDeque<>();
        idxQueue.add(new int[]{treeIdx});
        treeQueue.add(tree);

        while (!idxQueue.isEmpty()) {
            int nodeIdx = idxQueue.poll()[0];
            FasterTree node = treeQueue.poll();

            bfsAttributeIndex[nodeIdx] = node.getAttribute();
            bfsSplitPoint[nodeIdx] = node.getSplitPoint();

            FasterTree left = node.getSucessorLeft();
            FasterTree right = node.getSucessorRight();

            // Handle left child
            if (left.isLeaf()) {
                bfsChildLeft[nodeIdx] = -bfsPosClassProbs;
                bfsClassProbs[bfsPosClassProbs] = left.getClassProbs();
                bfsPosClassProbs++;
            } else {
                int leftIdx = bfsPosSplitNodes++;
                bfsChildLeft[nodeIdx] = leftIdx;
                idxQueue.add(new int[]{leftIdx});
                treeQueue.add(left);
            }

            // Handle right child
            if (right.isLeaf()) {
                bfsChildRight[nodeIdx] = -bfsPosClassProbs;
                bfsClassProbs[bfsPosClassProbs] = right.getClassProbs();
                bfsPosClassProbs++;
            } else {
                int rightIdx = bfsPosSplitNodes++;
                bfsChildRight[nodeIdx] = rightIdx;
                idxQueue.add(new int[]{rightIdx});
                treeQueue.add(right);
            }
        }
    }

}
