package cz.siret.prank.fforest.api;

import weka.core.Utils;

/**
 * FlatBinaryForest that remembers classProbabilities for both classes in each leaf
 * to accomodate for bugs in other RF implementations that could return probability >1 for some trees.
 */
public class LegacyFlatBinaryForest extends FlatBinaryForest {

    protected final double[][] classProbs;

    public LegacyFlatBinaryForest(int numTrees, int numAttributes, int[] childLeft, int[] childRight, int[] attributeIndex, double[] splitPoint, double[][] classProbs) {
        super(numTrees, numAttributes, childLeft, childRight, attributeIndex, splitPoint, null);
        this.classProbs = classProbs;
    }

//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        return predictClassProbs(instanceAttributes)[1];
    }



    @Override
    public double[] predictForBatch(double[][] instances) {
        int n = instances.length;
        double[] sumsClass0 = new double[n];
        double[] sumsClass1 = new double[n];


        for (int t=0; t!=numTrees; ++t) {
            for (int i=0; i!=n; ++i) {
                double[] classProbs = predictTreeClassProbs(t, instances[i]);
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

//===============================================================================================//

    public double[] predictClassProbs(double[] instanceAttributes) {
        double sum0 = 0d;
        double sum1 = 0d;

        for (int i=0; i!=numTrees; ++i) {
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

    @Override
    protected double predictTree(int tree, double[] instanceAttributes) {
        double[] probs = predictTreeClassProbs(tree, instanceAttributes);
        double p1 = probs[1];
        return p1 / (probs[0] + p1);
    }

}