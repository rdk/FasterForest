 package cz.siret.prank.ffutils;

import weka.core.Utils;

/**
 *
 */
public class NormalizationUtils {

    /**
     * Resuse second array and return it with normalizad probabilities
     * @param sumsClass0
     * @param sumsClass1
     * @return
     */
    public static double[] normalizedClass1ProbsReuseArray(double[] sumsClass0, double[] sumsClass1) {
        int n = sumsClass0.length;
        for (int i=0; i!=n; ++i) {
            sumsClass1[i] = normalizedProb(sumsClass0[i], sumsClass1[i]);
        }
        return sumsClass1;
    }

    /**
     * If sum is 0 returns 0.
     *
     * Note: doesn't handle NaNs
     */
    public static double normalizedProb(double p0, double p1) {
        p0 = p0 + p1; // reusing as sum

        if (p0 == 0.0) {
            return 0.0;
        }

        return p1 / p0;
    }


//===============================================================================================//

    /**
     * Normalizes the doubles in the array by their sum.
     *
     * If sum is 0 ignores.
     *
     * Note: doesn't handle NaNs
     */
    public static void normalizeBinary(double[] doubles) {

        double sum = doubles[0] + doubles[1];

        if (sum != 0.0) {
            doubles[0] /= sum;
            doubles[1] /= sum;
        }
    }

//===============================================================================================//


    public static double[] normalizedClass1Probs(double[] sumsClass0, double[] sumsClass1) {
        int n = sumsClass0.length;
        double[] res = new double[n];
        for (int i=0; i!=n; ++i) {
            double[] cp = new double[] { sumsClass0[i], sumsClass1[i] };
            Utils.normalize(cp);
            res[i] = cp[1];
        }
        return res;
    }


    public static void conditionallyEnsureNormalized(double[] classProbs, boolean ensureNormalized) {
        double sum = Utils.sum(classProbs);
        if (sum > 1d) {
            // System.out.println("Badly calibrated leaf class probs: " + Arrays.toString(classProbs));
            if (ensureNormalized) {
                Utils.normalize(classProbs, sum);
            }
        }
    }

    public static void conditionallyEnsureNormalizedBinary(float[] classProbs, boolean ensureNormalized) {
        float sum = classProbs[0] + classProbs[1];
        if (sum > 1f) {
            // System.out.println("Badly calibrated leaf class probs: " + Arrays.toString(classProbs));
            if (ensureNormalized) {
                normalizeBinaryFloats(classProbs, sum);
            }
        }
    }

    public static void normalizeBinaryFloats(float[] classProbs, float sum) {
        classProbs[0] /= sum;
        classProbs[1] /= sum;
    }
    
}
