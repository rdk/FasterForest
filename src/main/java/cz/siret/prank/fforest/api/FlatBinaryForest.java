package cz.siret.prank.fforest.api;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 *
 */
public class FlatBinaryForest implements BinaryForest, Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    private final int numTrees;
    private transient int maxDepth = -1; // lazy, -1 = not calculated yet

    private transient final double numTreesDouble;

    private final int[] childRight;
    private final int[] childLeft;
    private final int[] attributeIndex;
    private final double[] splitPoint;
    private final double[] score;

    @Override
    public int getNumClasses() {
        return 2;
    }

    @Override
    public int getNumTrees() {
        return numTrees;
    }

    @Override
    public int getMaxDepth() {
        if (maxDepth < 0) {
            maxDepth = calculateMaxDepth();
        }
        return maxDepth;
    }

    private int calculateMaxDepth() {
        int max = 0;

        for (int i=0; i!=numTrees; ++i) {
            max = Math.max(max, calculateTreeDepth(i));
        }

        return max;
    }

    @Override
    public double predict(double[] instanceAttributes) {
        double sum = 0d;

        for (int i=0; i!=numTrees; ++i) {
            sum += predictTree(i, instanceAttributes);
        }

        return sum / numTreesDouble;
    }

//===============================================================================================//

    private double predictTree(int tree, double[] instanceAttributes) {
        int currentNode = tree;
        int attr;

        while (true) {
            attr = attributeIndex[currentNode];

            if (instanceAttributes[attr] < splitPoint[currentNode]) {
                currentNode = childLeft[currentNode];
            } else {
                currentNode = childRight[currentNode];
            }

            if (currentNode < 0) {
                return score[-currentNode];
            }
        }
        
    }

    private int calculateTreeDepth(int tree) {
        if (tree < 0) {
            return 0;
        }

        int left = calculateTreeDepth(childLeft[tree]);
        int right = calculateTreeDepth(childRight[tree]);

        return Math.max(left, right) + 1;
    }

//===============================================================================================//


    public FlatBinaryForest(int numTrees, int[] childRight, int[] childLeft, int[] attributeIndex, double[] splitPoint, double[] score) {
        this.numTrees = numTrees;
        this.childRight = childRight;
        this.childLeft = childLeft;
        this.attributeIndex = attributeIndex;
        this.splitPoint = splitPoint;
        this.score = score;

        this.numTreesDouble = numTrees;
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
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
    
}
