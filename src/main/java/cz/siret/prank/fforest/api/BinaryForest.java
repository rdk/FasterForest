package cz.siret.prank.fforest.api;

import weka.core.Instance;

/**
 *
 */
public interface BinaryForest {

    int getNumTrees();

    int getMaxDepth();

    /**
     * Input vector length.
     */
    int getNumAttributes();

    /**
     * @param instanceAttributes length must be equal to getNumAttributes()
     */
    double predict(double[] instanceAttributes);

//===============================================================================================//

    default double[] distributionForInst(Instance instance) {
        double p = predict(instance.toDoubleArray());
        return new double[] {1d - p, p};
    }

    default int getNumClasses() {
        return 2;
    };

}
