package cz.siret.prank.fforest.api;

/**
 *
 */
public interface BinaryForest {

    int getNumTrees();

    int getMaxDepth();

    int getNumClasses();

    /**
     * Input vector length.
     */
    int getNumAttributes();

    /**
     * @param instanceAttributes length must be equal to getNumAttributes()
     */
    double predict(double[] instanceAttributes);

}
