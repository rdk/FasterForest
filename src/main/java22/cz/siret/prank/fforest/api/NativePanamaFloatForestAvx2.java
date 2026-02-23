package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Native float-precision forest prediction using AVX2 SIMD when available, with scalar fallback.
 *
 * <p>Identical to {@link NativePanamaFloatForest} except that batch prediction uses
 * the auto-dispatching native function which selects AVX2 on x86_64 CPUs that
 * support it (processing 8 float instances in parallel per tree traversal step).
 *
 * <p>Use {@link #isAvx2Available()} to check whether AVX2 is active.
 */
public class NativePanamaFloatForestAvx2 extends NativePanamaFloatForest {

    private NativePanamaFloatForestAvx2(int numTrees, int numAttributes, Arena arena,
                                        MemorySegment forestHandle) {
        super(numTrees, numAttributes, arena, forestHandle, FF_FLOAT_PREDICT_BATCH_AUTO);
    }

    /**
     * Returns true if the native library is loaded and AVX2 batch prediction is active.
     */
    public static boolean isAvx2Available() {
        return NATIVE_LOADED && NativePanamaForest.simdLevel() >= 2;
    }

    /**
     * Build from trained FasterTrees.
     *
     * @throws IllegalStateException if native library is not available
     */
    public static NativePanamaFloatForestAvx2 fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        if (!NATIVE_LOADED) {
            throw new IllegalStateException("Native library not available");
        }

        ContiguousDfsForest base = ContiguousDfsForest.fromFasterTrees(numAttributes, trees);
        return fromContiguousDfsForest(base);
    }

    /**
     * Build from an existing ContiguousDfsForest by converting double arrays to float
     * and copying to off-heap memory.
     */
    public static NativePanamaFloatForestAvx2 fromContiguousDfsForest(ContiguousDfsForest base) {
        if (!NATIVE_LOADED) {
            throw new IllegalStateException("Native library not available");
        }

        Arena arena = Arena.ofShared();
        try {
            MemorySegment handle = createFloatForestHandle(arena, base);
            return new NativePanamaFloatForestAvx2(base.numTrees, base.numAttributes, arena, handle);
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create native float AVX2 forest", t);
        }
    }
}
