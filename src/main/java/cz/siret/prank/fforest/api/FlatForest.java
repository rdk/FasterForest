package cz.siret.prank.fforest.api;

/**
 *
 */
public class FlatForest implements Forest {

    private int numClasses;
    private int numTrees;
    private transient int maxDepth = -1; // lazy, -1 = not calculated yet

    private int[] childRight;
    private int[] childLeft;
    private int[] attributeIndex;
    private double[] splitPoint;
    private double[][] distribution;

    @Override
    public int getNumTrees() {
        return numTrees;
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }


    @Override
    public int getMaxDepth() {
        if (maxDepth < 0) {
            maxDepth = calculateMaxDepth();
        }
        return maxDepth;
    }

    private int calculateMaxDepth() {
        // TODO
        return 0;
    }

    @Override
    public double[] predict(double[] instanceAttributes) {
        // TODO
        return new double[0];
    }

}
