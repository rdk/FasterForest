package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.List;

/**
 *
 */
public class FasterForestConverter {

    public enum ForestType {
        FlatBinaryForest,
        LegacyFlatBinaryForest,
        ShortFlatBinaryForest,
        SuperShortLegacyFlatBinaryForest,
        InterleavedBfsForest,
        InterleavedBfsDoubleForest,
        ContiguousBfsDoubleForest,
        SeparateArraysBfsForest,
        BranchlessBfsForest,
        ContiguousDfsForest,
        IlpDfsForest
    }

    public static BinaryForest convertFasterForest(TrainableFasterForest forest, ForestType targetType) {
        int numAttributes = forest.getNumAttributes();
        List<FasterTree> trees = forest.getTrees();

        switch (targetType) {
            case FlatBinaryForest:
                return FlatBinaryForestBuilder.buildFromFasterTrees(numAttributes, trees, false);
            case LegacyFlatBinaryForest:
                return FlatBinaryForestBuilder.buildFromFasterTreesLegacy(numAttributes, trees);
            case ShortFlatBinaryForest:
                return ShortLegacyFlatBinaryForest.from(FlatBinaryForestBuilder.buildFromFasterTreesLegacy(numAttributes, trees));
            case SuperShortLegacyFlatBinaryForest:
                return SuperShortLegacyFlatBinaryForest.from(FlatBinaryForestBuilder.buildFromFasterTreesLegacy(numAttributes, trees));
            case InterleavedBfsForest:
                return InterleavedBfsForest.fromFasterTrees(numAttributes, trees);
            case InterleavedBfsDoubleForest:
                return InterleavedBfsDoubleForest.fromFasterTrees(numAttributes, trees);
            case ContiguousBfsDoubleForest:
                return ContiguousBfsDoubleForest.fromFasterTrees(numAttributes, trees);
            case SeparateArraysBfsForest:
                return SeparateArraysBfsForest.fromFasterTrees(numAttributes, trees);
            case BranchlessBfsForest:
                return BranchlessBfsForest.fromFasterTrees(numAttributes, trees);
            case ContiguousDfsForest:
                return ContiguousDfsForest.fromFasterTrees(numAttributes, trees);
            case IlpDfsForest:
                return IlpDfsForest.fromFasterTrees(numAttributes, trees);
            default:
                throw new IllegalArgumentException("Unknown forest type: " + targetType);
        }
    }

}
