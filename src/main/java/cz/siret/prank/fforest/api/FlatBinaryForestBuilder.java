package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.Arrays;
import java.util.List;

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

}
