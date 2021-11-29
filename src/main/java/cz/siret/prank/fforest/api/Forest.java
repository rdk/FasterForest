package cz.siret.prank.fforest.api;

/**
 * Trained forest model
 */
public interface Forest {

    int getNumTrees();

    int getMaxDepth();

    int getNumClasses();

    double[] predict(double[] instanceAttributes);
    
}
