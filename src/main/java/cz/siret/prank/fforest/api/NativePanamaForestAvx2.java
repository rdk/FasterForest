package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.util.List;

/**
 * Stub for Java 17 compatibility. The real implementation lives in
 * {@code src/main/java22} and is loaded automatically on Java 22+ via
 * the multi-release JAR (META-INF/versions/22/).
 *
 * <p>On Java 17, {@link #isAvx2Available()} returns {@code false} and all
 * factory methods throw {@link UnsupportedOperationException}.
 */
public class NativePanamaForestAvx2 extends NativePanamaForest {

    protected NativePanamaForestAvx2() {
    }

    public static boolean isAvx2Available() {
        return false;
    }

    public static NativePanamaForestAvx2 fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        throw new UnsupportedOperationException("Native AVX2 forest requires Java 22+");
    }

    public static NativePanamaForestAvx2 fromContiguousDfsForest(ContiguousDfsForest base) {
        throw new UnsupportedOperationException("Native AVX2 forest requires Java 22+");
    }
}
