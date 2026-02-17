package cz.siret.prank.fforest.api;

import cz.siret.prank.fforest.FasterTree;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.List;

/**
 * Native C implementation of forest prediction using Panama FFM (Java 22+).
 *
 * <p>Delegates the hot prediction loop to a compiled C library that benefits from:
 * <ul>
 *   <li>No array bounds checks</li>
 *   <li>Branchless cmov for child selection</li>
 *   <li>AVX2 SIMD batch prediction (4 instances in parallel)</li>
 *   <li>Cache-line aligned data access</li>
 * </ul>
 *
 * <p>Data layout is identical to {@link ContiguousDfsForest} — separate int[]/double[]
 * arrays with contiguous per-tree DFS node ordering. The arrays are copied to off-heap
 * memory owned by an {@link Arena} and passed as raw pointers to the C library.
 *
 * <p>If the native library is not available on the current platform, use
 * {@link #isAvailable()} to check and fall back to {@link ContiguousDfsForest}.
 */
public class NativePanamaForest implements BinaryForest, Classifier, AutoCloseable {

    private static final boolean NATIVE_LOADED;

    // Method handles for native functions (resolved once at class load)
    private static final MethodHandle FF_FOREST_CREATE;
    private static final MethodHandle FF_FOREST_DESTROY;
    private static final MethodHandle FF_PREDICT;
    private static final MethodHandle FF_PREDICT_BATCH;
    private static final MethodHandle FF_SIMD_LEVEL;

    static {
        boolean loaded = false;
        MethodHandle create = null, destroy = null, predict = null, predictBatch = null, simdLevel = null;

        try {
            loaded = NativeLoader.load();
            if (loaded) {
                Linker linker = Linker.nativeLinker();
                SymbolLookup lookup = SymbolLookup.loaderLookup();

                create = linker.downcallHandle(
                        lookup.find("ff_forest_create").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS)
                );

                destroy = linker.downcallHandle(
                        lookup.find("ff_forest_destroy").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
                );

                predict = linker.downcallHandle(
                        lookup.find("ff_predict").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS)
                );

                predictBatch = linker.downcallHandle(
                        lookup.find("ff_predict_batch").orElseThrow(),
                        FunctionDescriptor.ofVoid(
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS)
                );

                simdLevel = linker.downcallHandle(
                        lookup.find("ff_simd_level").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT)
                );
            }
        } catch (Throwable t) {
            System.err.println("[FasterForest] Failed to initialize native bindings: " + t.getMessage());
            loaded = false;
        }

        NATIVE_LOADED = loaded;
        FF_FOREST_CREATE = create;
        FF_FOREST_DESTROY = destroy;
        FF_PREDICT = predict;
        FF_PREDICT_BATCH = predictBatch;
        FF_SIMD_LEVEL = simdLevel;
    }

//===============================================================================================//

    private final int numTrees;
    private final int numAttributes;
    private final Arena arena;
    private final MemorySegment forestHandle;

    // Pre-allocated reusable buffers for batch prediction (avoids per-call allocation)
    private MemorySegment instanceBuffer;
    private int instanceBufferCapacity;
    private MemorySegment outputBuffer;
    private int outputBufferCapacity;

    private NativePanamaForest(int numTrees, int numAttributes, Arena arena, MemorySegment forestHandle) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.arena = arena;
        this.forestHandle = forestHandle;
    }

//===============================================================================================//
// Static factory
//===============================================================================================//

    /**
     * Returns true if the native library is loaded and ready.
     */
    public static boolean isAvailable() {
        return NATIVE_LOADED;
    }

    /**
     * Returns the active SIMD level: 0=scalar, 2=AVX2, 3=AVX-512.
     */
    public static int simdLevel() {
        if (!NATIVE_LOADED) return -1;
        try {
            return (int) FF_SIMD_LEVEL.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /**
     * Build from trained FasterTrees. Uses the same data layout as
     * {@link ContiguousDfsForest}.
     *
     * @throws IllegalStateException if native library is not available
     */
    public static NativePanamaForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
        if (!NATIVE_LOADED) {
            throw new IllegalStateException("Native library not available");
        }

        // Build the flat arrays using ContiguousDfsForest's factory
        ContiguousDfsForest base = ContiguousDfsForest.fromFasterTrees(numAttributes, trees);

        return fromContiguousDfsForest(base);
    }

    /**
     * Build from an existing ContiguousDfsForest by copying its arrays to off-heap memory.
     */
    public static NativePanamaForest fromContiguousDfsForest(ContiguousDfsForest base) {
        if (!NATIVE_LOADED) {
            throw new IllegalStateException("Native library not available");
        }

        Arena arena = Arena.ofShared();
        try {
            // Copy Java arrays to off-heap memory
            MemorySegment treeRootsSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.treeRoots);
            MemorySegment childLeftSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.childLeft);
            MemorySegment childRightSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.childRight);
            MemorySegment attrIndexSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.attributeIndex);
            MemorySegment splitPointSeg = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, base.splitPoint);
            MemorySegment scoreSeg = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, base.score);

            int totalNodes = base.childLeft.length;
            int totalLeaves = base.score.length;

            // Create native forest handle
            MemorySegment handle = (MemorySegment) FF_FOREST_CREATE.invokeExact(
                    base.numTrees, base.numAttributes,
                    totalNodes, totalLeaves,
                    treeRootsSeg, childLeftSeg,
                    childRightSeg, attrIndexSeg,
                    splitPointSeg, scoreSeg
            );

            if (handle.equals(MemorySegment.NULL)) {
                arena.close();
                throw new RuntimeException("ff_forest_create returned NULL");
            }

            return new NativePanamaForest(base.numTrees, base.numAttributes, arena, handle);

        } catch (RuntimeException e) {
            arena.close();
            throw e;
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create native forest", t);
        }
    }

//===============================================================================================//

    @Override
    public int getNumAttributes() {
        return numAttributes;
    }

    @Override
    public int getNumTrees() {
        return numTrees;
    }

    @Override
    public int getMaxDepth() {
        return -1; // not tracked in native handle
    }

//===============================================================================================//
// Prediction
//===============================================================================================//

    @Override
    public double predict(double[] instanceAttributes) {
        try (Arena tempArena = Arena.ofConfined()) {
            MemorySegment instSeg = tempArena.allocateFrom(ValueLayout.JAVA_DOUBLE, instanceAttributes);
            return (double) FF_PREDICT.invokeExact(forestHandle, instSeg);
        } catch (Throwable t) {
            throw new RuntimeException("Native predict failed", t);
        }
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        final int n = instances.length;
        if (n == 0) return new double[0];

        try {
            final long rowBytes = (long) numAttributes * Double.BYTES;

            // Ensure pre-allocated buffers are large enough
            ensureBufferCapacity(n);

            // Flatten double[][] to contiguous off-heap buffer
            for (int i = 0; i < n; i++) {
                MemorySegment src = MemorySegment.ofArray(instances[i]);
                MemorySegment.copy(src, 0, instanceBuffer, (long) i * rowBytes, rowBytes);
            }

            // Call native batch prediction
            FF_PREDICT_BATCH.invokeExact(forestHandle, instanceBuffer, n, outputBuffer);

            // Copy results back to Java array
            double[] result = new double[n];
            MemorySegment dst = MemorySegment.ofArray(result);
            MemorySegment.copy(outputBuffer, 0, dst, 0, (long) n * Double.BYTES);
            return result;
        } catch (Throwable t) {
            throw new RuntimeException("Native predictForBatch failed", t);
        }
    }

    /**
     * Batch prediction from a contiguous off-heap buffer. Zero-copy path -- no data
     * marshalling overhead.
     *
     * <p>The data segment must contain {@code n} rows of {@code numAttributes} doubles
     * in row-major order (row 0 attr 0, row 0 attr 1, ..., row 1 attr 0, ...).
     * Total size must be at least {@code n * numAttributes * Double.BYTES} bytes.
     *
     * @param data contiguous off-heap MemorySegment with instance data (row-major doubles)
     * @param n    number of instances (rows) in the data segment
     * @return prediction scores (one per instance)
     */
    public double[] predictForBatchContiguous(MemorySegment data, int n) {
        if (n == 0) return new double[0];

        try {
            ensureOutputCapacity(n);

            FF_PREDICT_BATCH.invokeExact(forestHandle, data, n, outputBuffer);

            double[] result = new double[n];
            MemorySegment dst = MemorySegment.ofArray(result);
            MemorySegment.copy(outputBuffer, 0, dst, 0, (long) n * Double.BYTES);
            return result;
        } catch (Throwable t) {
            throw new RuntimeException("Native predictForBatchContiguous failed", t);
        }
    }

    /**
     * Flatten a {@code double[][]} into a contiguous off-heap MemorySegment suitable for
     * {@link #predictForBatchContiguous(MemorySegment, int)}.
     *
     * <p>The returned segment is allocated in the given arena and lives until that arena
     * is closed. Callers who reuse the same instances across multiple predictions should
     * flatten once and call {@code predictForBatchContiguous} repeatedly.
     *
     * @param instances Java double[][] array (each row has numAttributes elements)
     * @param targetArena arena that owns the returned segment
     * @return contiguous row-major off-heap segment
     */
    public static MemorySegment flattenToOffHeap(double[][] instances, int numAttributes, Arena targetArena) {
        final int n = instances.length;
        final long rowBytes = (long) numAttributes * Double.BYTES;
        MemorySegment seg = targetArena.allocate(n * rowBytes, Double.BYTES);
        for (int i = 0; i < n; i++) {
            MemorySegment src = MemorySegment.ofArray(instances[i]);
            MemorySegment.copy(src, 0, seg, (long) i * rowBytes, rowBytes);
        }
        return seg;
    }

    private void ensureBufferCapacity(int n) {
        if (n > instanceBufferCapacity) {
            instanceBuffer = arena.allocate((long) n * numAttributes * Double.BYTES, Double.BYTES);
            instanceBufferCapacity = n;
        }
        ensureOutputCapacity(n);
    }

    private void ensureOutputCapacity(int n) {
        if (n > outputBufferCapacity) {
            outputBuffer = arena.allocate((long) n * Double.BYTES, Double.BYTES);
            outputBufferCapacity = n;
        }
    }

//===============================================================================================//

    @Override
    public void close() {
        try {
            FF_FOREST_DESTROY.invokeExact(forestHandle);
        } catch (Throwable t) {
            // ignore
        }
        arena.close();
    }

//===============================================================================================//
// Classifier interface (for Weka compatibility)
//===============================================================================================//

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // do nothing
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return distributionForInst(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
