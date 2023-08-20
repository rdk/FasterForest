package cz.siret.prank.ffutils;

import weka.core.Utils;

/**
 *
 */
public class NormalizationUtils {

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
