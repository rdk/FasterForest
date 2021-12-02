package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.List;

/**
 *
 */
public class FlatBinaryForestBuilder {

    int[] childRight;
    int[] childLeft;
    int[] attributeIndex;
    double[] splitPoint;
    double[] score;

    int posSplitNodes = 0;
    int posScore = 1;


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
            score[posScore] = getScoreFromProbs(tree);
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
            score[posScore] = getScoreFromProbs(left);
            posScore++;
        } else {
            leftIdx = posSplitNodes++;
            childLeft[treeIdx] = leftIdx;
        }

        if (right.isLeaf()) {
            childRight[treeIdx] = -posScore;
            score[posScore] = getScoreFromProbs(right);
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

    private double getScoreFromProbs(FasterTree tree) {
        double[] hist = tree.getClassProbs();
        return hist[1] / (hist[0] + hist[1]);
    }

}
