package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * Lightweight wrapper around a list of FasterTree objects.
 * Implements TrainableFasterForest so it can be passed to FasterForestConverter
 * for conversion to any flat forest format.
 *
 * Not trainable — buildClassifier throws UnsupportedOperationException.
 */
public class FasterTreeForest implements TrainableFasterForest, Serializable {

    private static final long serialVersionUID = 1L;

    private final int numAttributes;
    private final List<FasterTree> trees;

    public FasterTreeForest(int numAttributes, List<FasterTree> trees) {
        this.numAttributes = numAttributes;
        this.trees = Collections.unmodifiableList(trees);
    }

    @Override
    public int getNumAttributes() {
        return numAttributes;
    }

    @Override
    public List<FasterTree> getTrees() {
        return trees;
    }

    @Override
    public void buildClassifier(Instances data) {
        throw new UnsupportedOperationException("FasterTreeForest is not trainable");
    }

}
