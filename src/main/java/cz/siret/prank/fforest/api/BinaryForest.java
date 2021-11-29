package cz.siret.prank.fforest.api;

/**
 *
 */
public interface BinaryForest {

    int getNumTrees();

    int getMaxDepth();

    int getNumClasses();

    double predict(double[] instanceAttributes);

}
