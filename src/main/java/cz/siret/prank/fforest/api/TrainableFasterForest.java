package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.core.Instances;

import java.util.List;

/**
 *
 */
public interface TrainableFasterForest {

    int getNumAttributes();

    List<FasterTree> getTrees();

    void buildClassifier(Instances data) throws Exception;

}
