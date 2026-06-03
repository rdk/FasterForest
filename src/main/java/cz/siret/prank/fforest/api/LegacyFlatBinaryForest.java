package cz.siret.prank.fforest.api;

import cz.siret.prank.ffutils.NormalizationUtils;

import java.io.Serial;

import static cz.siret.prank.ffutils.NormalizationUtils.normalizedClass1ProbsReuseArray;

/**
 * The <b>faithful</b> flat forest: it is the reference / recommended default, <b>not</b> deprecated
 * despite the "Legacy" name. It remembers the class probabilities of both classes in each leaf and
 * aggregates them <i>sum-then-normalize</i>, so it reproduces the trained {@link FasterForest}'s
 * predictions exactly (to 1e-15). This matters because trained leaves generally do not sum to 1 (they
 * encode mean bootstrap multiplicity); the scalar-{@code score} variants pre-normalize each leaf and so
 * can diverge from the trained model. Use this (or {@link ShortLegacyFlatBinaryForest}) when exact
 * agreement with the trained model is required. See {@code PREDICTION-SEMANTICS.md} and {@code VARIANTS.md}.
 *
 * <p>(The dual-class storage also accommodates other RF implementations that can return a per-tree
 * probability &gt; 1.)
 */
public class LegacyFlatBinaryForest extends FlatBinaryForest {

    @Serial
    private static final long serialVersionUID = -4570003757601764377L;

    protected final double[][] classProbs;

    public LegacyFlatBinaryForest(int numTrees, int numAttributes, int[] childLeft, int[] childRight, int[] attributeIndex, double[] splitPoint, double[][] classProbs) {
        super(numTrees, numAttributes, childLeft, childRight, attributeIndex, splitPoint, null);
        this.classProbs = classProbs;
    }

//===============================================================================================//

    @Override
    public double predict(final double[] instanceAttributes) {
        return predictClassProbs(instanceAttributes)[1];
    }

    @Override
    public double[] predictForBatch(final double[][] instances) {
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

        return normalizedClass1ProbsReuseArray(sumsClass0, sumsClass1);
    }

//===============================================================================================//

    public double[] predictClassProbs(final double[] instanceAttributes) {
        double sum0 = 0d;
        double sum1 = 0d;

        for (int i=0; i!=numTrees; ++i) {
            double[] probs = predictTreeClassProbs(i, instanceAttributes);
            sum0 += probs[0];
            sum1 += probs[1];
        }

        double[] res = new double[] { sum0, sum1 };
        NormalizationUtils.normalizeBinary(res);
        return res;
    }

    /**
     * @param currentNode = tree number when called for the first time for a tree
     * @param instanceAttributes
     * @return
     */
    protected double[] predictTreeClassProbs(int currentNode, final double[] instanceAttributes) {
        final int[] childRight = this.childRight;
        final int[] childLeft = this.childLeft;
        final int[] attributeIndex = this.attributeIndex;
        final double[] splitPoint = this.splitPoint;

        do {
            if (instanceAttributes[attributeIndex[currentNode]] < splitPoint[currentNode]) {
                currentNode = childLeft[currentNode];
            } else {
                currentNode = childRight[currentNode];
            }

            if (currentNode < 0) {
                return classProbs[-currentNode];
            }
        } while (true);

    }

    @Override
    protected double predictTree(int tree, double[] instanceAttributes) {
        double[] probs = predictTreeClassProbs(tree, instanceAttributes);
        double p1 = probs[1];
        return p1 / (probs[0] + p1);
    }

}
