package cz.siret.prank.fforest.api;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.Serializable;
import java.util.Arrays;

/**
 *
 */
public class FlatBinaryForest implements BinaryForest, Classifier, Serializable {

    private static final long serialVersionUID = 1L;

    protected final int numTrees;
    protected final int numAttributes;
    protected final int[] childRight;
    protected final int[] childLeft;
    protected final int[] attributeIndex;
    protected final double[] splitPoint;
    protected final double[] score;



    protected transient final double numTreesAsDouble;  // cache to avoid repeated type conversion
    protected transient int maxDepth = -1; // lazy, -1 = not calculated yet

    protected transient int[] treeDepths;


//===============================================================================================//

    public FlatBinaryForest(int numTrees, int numAttributes, int[] childLeft, int[] childRight, int[] attributeIndex, double[] splitPoint, double[] score) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.childLeft = childLeft;
        this.childRight = childRight;
        this.attributeIndex = attributeIndex;
        this.splitPoint = splitPoint;
        this.score = score;

        this.numTreesAsDouble = numTrees;
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
        if (maxDepth < 0) {
            calculateDepths();
        }
        return maxDepth;
    }

    public int[] getTreeDepths() {
        if (treeDepths == null) {
            calculateDepths();
        }
        return treeDepths;
    }

//===============================================================================================//

    private void calculateDepths() {
        treeDepths = calculateTreeDepths();
        maxDepth = Arrays.stream(treeDepths).max().getAsInt();
    }

    private int[] calculateTreeDepths() {
        int[] depths = new int[numTrees];
        for (int i=0; i!=numTrees; ++i) {
            depths[i] = calculateTreeDepth(i);
        }
        return depths;
    }

    private int calculateMaxDepth() {
        int max = 0;

        for (int i=0; i!=numTrees; ++i) {
            max = Math.max(max, calculateTreeDepth(i));
        }

        return max;
    }


    /**
     * @return max tree depth
     */
    private int calculateTreeDepth(int tree) {
        if (tree < 0) {
            return 1;
        }

        int left = calculateTreeDepth(childLeft[tree]);
        int right = calculateTreeDepth(childRight[tree]);

        return Math.max(left, right) + 1;
    }


//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        double sum = 0d;

        for (int i=0; i!=numTrees; ++i) {
            sum += predictTree(i, instanceAttributes);
        }

        return sum / numTreesAsDouble;
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        int n = instances.length;
        double[] sums = new double[n];

        for (int t=0; t!=numTrees; ++t) {
            for (int i=0; i!=n; ++i) {
                sums[i] = predictTree(t, instances[i]);
            }
        }

        for (int i=0; i!=n; ++i) {
            sums[i] /= numTrees;
        }
        return sums;
    }

//===============================================================================================//

    protected double predictTree(int tree, double[] instanceAttributes) {
        int currentNode = tree;
        int attr;

        // assumes that tree is not a leaf
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

    public double[] evalTrees(double[] instanceAttributes) {
        double[] res = new double[numTrees];
        for (int i=0; i!=numTrees; ++i) {
            res[i] = predictTree(i, instanceAttributes);
        }
        return res;
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
