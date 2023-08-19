package cz.siret.prank.fforest.api;

import weka.core.Utils;

import java.util.*;
import java.util.stream.Collectors;

/**
 *
 */
public class OptimizingFlatBinaryForest implements BinaryForest {

    LegacyFlatBinaryForest forest;

    long[] treePositionCounts;
    long[] scorePositionCounts;


    List<NodeInfo> splitNodes;
    List<NodeInfo> leafNodes;

    public OptimizingFlatBinaryForest(LegacyFlatBinaryForest forest) {
        this.forest = forest;

        treePositionCounts = new long[forest.splitPoint.length];
        scorePositionCounts = new long[forest.classProbs.length];
    }

//===============================================================================================//

    public static class NodeOrderings {
        public static Comparator<NodeInfo> BY_TREE = Comparator.comparingInt(NodeInfo::getTree);
        public static Comparator<NodeInfo> BY_COUNT = Comparator.comparingLong(n -> -n.count);
        public static Comparator<NodeInfo> BY_TREE_COUNT = Comparator.comparingInt(NodeInfo::getTree).thenComparingLong(n -> -n.count);
        public static Comparator<NodeInfo> BY_TREE_DEPTH_COUNT = Comparator.comparingInt(NodeInfo::getTree).thenComparingInt(NodeInfo::getDepth).thenComparingLong(n -> -n.count);
        public static Comparator<NodeInfo> BY_TREE_DEPTH = Comparator.comparingInt(NodeInfo::getTree).thenComparingInt(NodeInfo::getDepth);
        public static Comparator<NodeInfo> BY_DEPTH = Comparator.comparingInt(NodeInfo::getDepth);
        public static Comparator<NodeInfo> BY_DEPTH_COUNT = Comparator.comparingInt(NodeInfo::getDepth).thenComparingLong(n -> -n.count);

        public static Comparator<NodeInfo> DEFAULT = BY_TREE;
    }

    public LegacyFlatBinaryForest buildOptimizedForest() {
        return buildOptimizedForest(NodeOrderings.DEFAULT, NodeOrderings.DEFAULT);
    }

    public LegacyFlatBinaryForest buildOptimizedForest(Comparator<NodeInfo> splitNodeOrdering, Comparator<NodeInfo> leafNodeOrdering) {
        splitNodes = countsToPosCounts(treePositionCounts);
        leafNodes = countsToPosCounts(scorePositionCounts);
        labelSplitNodesByTreesAndDepth(splitNodes, leafNodes, forest);

        splitNodes = reorderSplitNodes(splitNodes, splitNodeOrdering);
        leafNodes = leafNodes.subList(1, leafNodes.size());
        leafNodes.sort(leafNodeOrdering); // stable sort


        return reorderForest(forest, splitNodes, leafNodes);
    }

    private List<NodeInfo> reorderSplitNodes(List<NodeInfo> splitNodes, Comparator<NodeInfo> comparator) {
        List<NodeInfo> head = splitNodes.subList(0, getNumTrees());
        List<NodeInfo> toSort = new ArrayList<>(splitNodes.subList(getNumTrees(), splitNodes.size()));

        List<NodeInfo> res = new ArrayList<>(head);
        toSort.sort(comparator);
        res.addAll(toSort);
        
        return res;
    }

    List<List<NodeInfo>> splitByTrees(List<NodeInfo> splitNodes, int numTrees) {
        Map<Integer, List<NodeInfo>> nodesByTree = splitNodes.stream().collect(Collectors.groupingBy(NodeInfo::getTree));

        List<List<NodeInfo>> res = new ArrayList<>(numTrees);
        for (int i=0; i!=numTrees; ++i) {
            res.add(new ArrayList<>(nodesByTree.get(i)));
        }

        return res;
    }

    /**
     * Reorder arrays based on visit count.
     *
     * @param treePositions  sorted by count DESC
     * @param scorePositions  sorted by count DESC
     */
    private LegacyFlatBinaryForest reorderForest(LegacyFlatBinaryForest f, List<NodeInfo> treePositions, List<NodeInfo> scorePositions) {
        int numSplits = f.splitPoint.length;
        int numLeaves = f.classProbs.length - 1;

        int[] childLeft = new int[numSplits];
        int[] childRight = new int[numSplits];
        int[] attributeIndex = new int[numSplits];
        double[] splitPoint = new double[numSplits];
        double[][] classProbs = new double[numLeaves + 1][];

        // build old to new pos mappings
        int[] treePosOldToNew = new int[numSplits];
        int[] scorePosOldToNew = new int[numLeaves + 1];
        for (int i=0; i!=numSplits; ++i) {
            int oldPos = treePositions.get(i).position;
            treePosOldToNew[oldPos] = i;
        }
        for (int i=0; i!=numLeaves; ++i) {
            int oldPos = scorePositions.get(i).position;
            scorePosOldToNew[oldPos] = i + 1;  // skipping 0 by design **
        }
        //System.out.println("treePosOldToNew:" + Arrays.toString(treePosOldToNew));
        //System.out.println("scorePosOldToNew:" + Arrays.toString(scorePosOldToNew));

        for (int i=0; i!=numSplits; ++i) {
            // newPos = i;
            int oldPos = treePositions.get(i).position;

            int oldChildLeft = f.childLeft[oldPos];
            int oldChildRight = f.childRight[oldPos];
            int newChildLeft = (oldChildLeft >= 0) ? treePosOldToNew[oldChildLeft] : -scorePosOldToNew[-oldChildLeft];
            int newChildRight = (oldChildRight >= 0) ? treePosOldToNew[oldChildRight] : -scorePosOldToNew[-oldChildRight];
            childLeft[i] = newChildLeft;
            childRight[i] = newChildRight;

            attributeIndex[i] = f.attributeIndex[oldPos];
            splitPoint[i] = f.splitPoint[oldPos];
        }
        for (int i=0; i!=numLeaves; ++i) {
            // newPos = i+1;
            int oldPos = scorePositions.get(i).position;
            classProbs[i+1] = f.classProbs[oldPos];       // skipping classProbs[0] by design **
        }

        return new LegacyFlatBinaryForest(f.numTrees, f.numAttributes, childLeft, childRight, attributeIndex, splitPoint, classProbs);
    }


    private List<NodeInfo> countsToPosCounts(long[] counts) {
        int n = counts.length;
        List<NodeInfo> posCounts = new ArrayList<>(n);
        for (int i=0; i!=n; ++i) {
            posCounts.add(new NodeInfo(i, counts[i]));
        }
        return posCounts;
    }

    void labelSplitNodesByTreesAndDepth(List<NodeInfo> splitNodes, List<NodeInfo> scoreNodes, LegacyFlatBinaryForest forest) {
        for (int i=0; i!=forest.numTrees; ++i) {
            labelSplitNodes(i, i, 0, splitNodes, scoreNodes, forest);
        }
    }
    
    void labelSplitNodes(int currPos, int tree, int depth, List<NodeInfo> splitNodes, List<NodeInfo> scoreNodes, LegacyFlatBinaryForest forest) {
        if (currPos < 0) { // leaf
            scoreNodes.get(-currPos).tree = tree;
            scoreNodes.get(-currPos).depth = depth;
        } else {  // split node
            splitNodes.get(currPos).tree = tree;
            splitNodes.get(currPos).depth = depth;
            labelSplitNodes(forest.childLeft[currPos], tree, depth+1, splitNodes, scoreNodes, forest);
            labelSplitNodes(forest.childRight[currPos], tree, depth+1, splitNodes, scoreNodes, forest);
        }
    }


    public static class NodeInfo {
        int position;
        long count;

        int tree;

        int depth;

        public NodeInfo(int position, long count) {
            this.position = position;
            this.count = count;
        }

        public NodeInfo(int position, long count, int tree, int depth) {
            this.position = position;
            this.count = count;
            this.tree = tree;
            this.depth = depth;
        }

        public int getPosition() {
            return position;
        }

        public void setPosition(int position) {
            this.position = position;
        }

        public long getCount() {
            return count;
        }

        public void setCount(long count) {
            this.count = count;
        }

        public int getTree() {
            return tree;
        }

        public void setTree(int tree) {
            this.tree = tree;
        }

        public int getDepth() {
            return depth;
        }

        public void setDepth(int depth) {
            this.depth = depth;
        }

        @Override
        public String toString() {
            return "[" + position + "," + count + "]";
        }
    }


//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        return predictClassProbs(instanceAttributes)[1];
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        throw new UnsupportedOperationException();
    }

    public double[] predictClassProbs(double[] instanceAttributes) {
        double sum0 = 0d;
        double sum1 = 0d;

        for (int i=0; i!=forest.numTrees; ++i) {
            double[] probs = predictTreeClassProbs(i, instanceAttributes);
            sum0 += probs[0];
            sum1 += probs[1];
        }

        double[] res = new double[] { sum0, sum1 };
        Utils.normalize(res);
        return res;
    }


    protected double[] predictTreeClassProbs(int tree, double[] instanceAttributes) {
        int currentNode = tree;
        int attr;

        treePositionCounts[tree]++; // count

        while (true) {
            attr = forest.attributeIndex[currentNode];

            if (instanceAttributes[attr] < forest.splitPoint[currentNode]) {
                currentNode = forest.childLeft[currentNode];
            } else {
                currentNode = forest.childRight[currentNode];
            }

            if (currentNode < 0) {
                scorePositionCounts[-currentNode]++;   // count
                return forest.classProbs[-currentNode];
            } else {
                treePositionCounts[currentNode]++; // count
            }
        }

    }

//===============================================================================================//

    @Override
    public int getNumTrees() {
        return forest.getNumTrees();
    }

    @Override
    public int getMaxDepth() {
        return forest.getMaxDepth();
    }

    @Override
    public int getNumClasses() {
        return forest.getNumClasses();
    }

    @Override
    public int getNumAttributes() {
        return forest.getNumAttributes();
    }

//===============================================================================================//

    public List<NodeInfo> getSplitNodes() {
        return splitNodes;
    }

    public List<NodeInfo> getLeafNodes() {
        return leafNodes;
    }

}
