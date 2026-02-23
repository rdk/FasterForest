package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

/**
 * Converts a trained Weka {@link RandomForest} into a {@link FasterTreeForest}.
 *
 * Only supports forests trained on all-numeric attributes with a nominal class.
 * Missing values and nominal attribute splits are not supported.
 *
 * The resulting {@link FasterTreeForest} can be passed to
 * {@link FasterForestConverter#convertFasterForest} for conversion to any optimized forest format.
 */
public class WekaRandomForestConverter {

    // Cached reflection fields for RandomTree inner Tree class
    private final Field treeNodeAttribute;
    private final Field treeNodeSplitPoint;
    private final Field treeNodeSuccessors;
    private final Field treeNodeClassDistribution;

    private WekaRandomForestConverter(
        Field treeNodeAttribute,
        Field treeNodeSplitPoint,
        Field treeNodeSuccessors,
        Field treeNodeClassDistribution
    ) {
        this.treeNodeAttribute = treeNodeAttribute;
        this.treeNodeSplitPoint = treeNodeSplitPoint;
        this.treeNodeSuccessors = treeNodeSuccessors;
        this.treeNodeClassDistribution = treeNodeClassDistribution;
    }

    /**
     * Converts a trained Weka {@link RandomForest} into a {@link FasterTreeForest}.
     *
     * @param wekaForest a trained Weka RandomForest (all-numeric attributes, nominal class)
     * @return FasterTreeForest that can be used for prediction or further conversion
     * @throws IllegalArgumentException if the forest is not trained, contains nominal splits,
     *                                  or has degenerate trees
     */
    public static FasterTreeForest toFasterTreeForest(RandomForest wekaForest) {
        try {
            return doConvert(wekaForest);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Failed to convert Weka RandomForest", e);
        }
    }

    private static FasterTreeForest doConvert(RandomForest wekaForest) throws Exception {
        // Reflect m_Classifiers from IteratedSingleClassifierEnhancer
        Field classifiersField = IteratedSingleClassifierEnhancer.class.getDeclaredField("m_Classifiers");
        classifiersField.setAccessible(true);
        Classifier[] classifiers = (Classifier[]) classifiersField.get(wekaForest);

        if (classifiers == null || classifiers.length == 0) {
            throw new IllegalArgumentException("RandomForest has not been trained (no classifiers)");
        }

        // Reflect m_Tree and m_Info from RandomTree
        Field mTreeField = RandomTree.class.getDeclaredField("m_Tree");
        mTreeField.setAccessible(true);
        Field mInfoField = RandomTree.class.getDeclaredField("m_Info");
        mInfoField.setAccessible(true);

        // Get first tree to resolve inner Tree class and numAttributes
        RandomTree firstRandomTree = (RandomTree) classifiers[0];
        Object firstTreeNode = mTreeField.get(firstRandomTree);
        if (firstTreeNode == null) {
            throw new IllegalArgumentException(
                "RandomTree at index 0 has no tree structure (ZeroR fallback). Cannot convert.");
        }

        Instances info = (Instances) mInfoField.get(firstRandomTree);
        int numAttributes = info.numAttributes(); // includes class attribute

        // Cache inner Tree class field accessors
        Class<?> treeNodeClass = firstTreeNode.getClass();
        Field attrField = treeNodeClass.getDeclaredField("m_Attribute");
        attrField.setAccessible(true);
        Field splitField = treeNodeClass.getDeclaredField("m_SplitPoint");
        splitField.setAccessible(true);
        Field successorsField = treeNodeClass.getDeclaredField("m_Successors");
        successorsField.setAccessible(true);
        Field classDistField = treeNodeClass.getDeclaredField("m_ClassDistribution");
        classDistField.setAccessible(true);

        WekaRandomForestConverter converter = new WekaRandomForestConverter(
            attrField, splitField, successorsField, classDistField
        );

        // Convert all trees
        List<FasterTree> trees = new ArrayList<>(classifiers.length);
        for (int i = 0; i < classifiers.length; i++) {
            RandomTree rt = (RandomTree) classifiers[i];
            Object treeNode = mTreeField.get(rt);
            if (treeNode == null) {
                throw new IllegalArgumentException(
                    "RandomTree at index " + i + " has no tree structure (ZeroR fallback). Cannot convert.");
            }
            trees.add(converter.convertNode(treeNode, i));
        }

        return new FasterTreeForest(numAttributes, trees);
    }

    private FasterTree convertNode(Object node, int treeIndex) throws Exception {
        int attribute = treeNodeAttribute.getInt(node);

        if (attribute == -1) {
            // Leaf node
            double[] classDist = (double[]) treeNodeClassDistribution.get(node);
            if (classDist == null) {
                throw new IllegalArgumentException(
                    "Tree " + treeIndex + ": leaf node has null class distribution (empty leaf)");
            }

            // Weka stores raw weighted class counts in m_ClassDistribution and normalizes
            // at prediction time (Utils.normalize). We must pre-normalize here because
            // FasterTree returns m_ClassProbs directly without normalization. Without this,
            // trees with larger leaves would dominate the ensemble sum before final normalization,
            // effectively weighting trees by leaf size — diverging from Weka's predictions.
            double[] classProbs = new double[classDist.length];
            double sum = 0;
            for (double v : classDist) {
                sum += v;
            }
            if (sum <= 0) {
                throw new IllegalArgumentException(
                    "Tree " + treeIndex + ": leaf node has zero-sum class distribution");
            }
            for (int c = 0; c < classDist.length; c++) {
                classProbs[c] = classDist[c] / sum;
            }

            return new FasterTree(null, null, -1, Double.NaN, classProbs);
        } else {
            // Split node
            double splitPoint = treeNodeSplitPoint.getDouble(node);
            Object[] successors = (Object[]) treeNodeSuccessors.get(node);

            if (successors.length != 2) {
                throw new IllegalArgumentException(
                    "Tree " + treeIndex + ": attribute " + attribute
                        + " has " + successors.length + " successors (expected 2 for numeric split)."
                        + " Nominal attribute splits are not supported.");
            }

            FasterTree left = convertNode(successors[0], treeIndex);
            FasterTree right = convertNode(successors[1], treeIndex);

            return new FasterTree(left, right, attribute, splitPoint, null);
        }
    }

}
