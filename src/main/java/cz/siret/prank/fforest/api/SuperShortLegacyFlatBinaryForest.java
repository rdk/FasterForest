package cz.siret.prank.fforest.api;

import weka.core.Utils;

/**
 * Like ShortLegacyFlatBinaryForest, but with short instead of int for tree structure.
 * This is more memory efficient, but can only be used for small trees (max 32767 nodes per tree).
 */
public class SuperShortLegacyFlatBinaryForest implements BinaryForest {

    private final int numTrees;
    private final int numAttributes;
    private final short[] childRight;
    private final short[] childLeft;
    private final short[] attributeIndex;
    private final float[] splitPoint;

    private final float[][] classProbs;


    private transient final double numTreesAsDouble;  // cache to avoid repeated type conversion
    private transient int maxDepth = -1; // lazy, -1 = not calculated yet

//===============================================================================================//

    public SuperShortLegacyFlatBinaryForest(int numTrees, int numAttributes, short[] childLeft, short[] childRight, short[] attributeIndex, float[] splitPoint, float[][] classProbs) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.childLeft = childLeft;
        this.childRight = childRight;
        this.attributeIndex = attributeIndex;
        this.splitPoint = splitPoint;
        this.classProbs = classProbs;

        this.numTreesAsDouble = numTrees;
    }

    public static SuperShortLegacyFlatBinaryForest from(LegacyFlatBinaryForest forest) {
        float[] splitPoint = new float[forest.splitPoint.length];
        for (int i=0; i!=splitPoint.length; ++i) {
            splitPoint[i] = (float) forest.splitPoint[i];
        }
        float[][] classProbs = new float[forest.classProbs.length][];
        for (int i=1; i!=classProbs.length; ++i) {
            classProbs[i] = new float[] {(float)forest.classProbs[i][0], (float)forest.classProbs[i][1]};
        }

        short[] childLeft = intsToShorts(forest.childLeft);
        short[] childRight = intsToShorts(forest.childRight);
        short[] attributeIndex = intsToShorts(forest.attributeIndex);

        return new SuperShortLegacyFlatBinaryForest(forest.numTrees, forest.numAttributes, childLeft, childRight, attributeIndex, splitPoint, classProbs);
    }

    public static short[] intsToShorts(int[] ints) {
        int n = ints.length;
        short[] res = new short[n];
        for (int i=0; i!=n; ++i) {
            res[i] = (short) ints[i];
        }
        return res;
    }


//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        return predictClassProbs(instanceAttributes)[1];
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        int n = instances.length;
        float[] sumsClass0 = new float[n];
        float[] sumsClass1 = new float[n];


        for (int t=0; t!=numTrees; ++t) {
            for (int i=0; i!=n; ++i) {
                float[] classProbs = predictTreeClassProbs(t, instances[i]);
                sumsClass0[i] += classProbs[0];
                sumsClass1[i] += classProbs[1];
            }
        }

        double[] res = new double[n];
        for (int i=0; i!=n; ++i) {
            double[] cp = new double[] { sumsClass0[i], sumsClass1[i] };
            Utils.normalize(cp);
            res[i] = cp[1];
        }
        return res;
    }

    public double[] predictClassProbs(double[] instanceAttributes) {
        float sum0 = 0f;
        float sum1 = 0f;

        for (int i=0; i!=numTrees; ++i) {
            float[] probs = predictTreeClassProbs(i, instanceAttributes);
            sum0 += probs[0];
            sum1 += probs[1];
        }

        double[] res = new double[] { sum0, sum1 };
        Utils.normalize(res);
        return res;
    }

    protected float[] predictTreeClassProbs(int tree, double[] instanceAttributes) {
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
                return classProbs[-currentNode];
            }
        }

    }


//===============================================================================================//

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
    public int getNumTrees() {
        return numTrees;
    }

    @Override
    public int getMaxDepth() {
        return calculateMaxDepth();
    }

    @Override
    public int getNumAttributes() {
        return numAttributes;
    }

    
}
