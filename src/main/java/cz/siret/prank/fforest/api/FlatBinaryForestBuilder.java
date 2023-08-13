package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.List;

/**
 *
 */
public class FlatBinaryForestBuilder {

    /**
     * Use only positive class probability
     */
    boolean useOnlyPositive = false;


//===============================================================================================//

    int[] childRight;
    int[] childLeft;
    int[] attributeIndex;
    double[] splitPoint;
    double[] score;

    int posSplitNodes = 0;
    int posScore = 1;

//===============================================================================================//

    /**
     * @param trees
     * @param useOnlyPositiveClassProbability Use only positive class probability p_class[1] instead of ratio p_class[1] / (p_class[0] + p_class[1])
     */
    public FlatBinaryForest buildFromFasterTrees(List<FasterTree> trees, boolean useOnlyPositiveClassProbability) {
        useOnlyPositive = useOnlyPositiveClassProbability;
        return buildFromFasterTrees(trees);
    }


    public FlatBinaryForest buildFromFasterTrees(List<FasterTree> trees) {

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
        score = new double[leaves+1];

        posSplitNodes = numTrees; // leave first n as root nodes for each tree

        for (int i=0; i!=numTrees; ++i) {
            compileTree(i, trees.get(i));
        }

        return new FlatBinaryForest(trees.size(), childRight, childLeft, attributeIndex, splitPoint, score);
    }

    private void compileTree(int treeIdx, FasterTree tree) {
        if (tree.isLeaf()) {
            childLeft[treeIdx] = -posScore;
            childRight[treeIdx] = -posScore;
            score[posScore] = getScoreFromProbs(tree.getClassProbs());
            posScore++;
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
            childLeft[treeIdx] = -posScore;
            score[posScore] = getScoreFromProbs(left.getClassProbs());
            posScore++;
        } else {
            leftIdx = posSplitNodes++;
            childLeft[treeIdx] = leftIdx;
        }

        if (right.isLeaf()) {
            childRight[treeIdx] = -posScore;
            score[posScore] = getScoreFromProbs(right.getClassProbs());
            posScore++;
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
        if (classProbs == null) {
            throw new RuntimeException("classProbsis null");
        }

        double p1 = classProbs[1];

        if (useOnlyPositive) {
            return p1;
        } else {
            return p1 / (classProbs[0] + p1);
        }
    }

}
