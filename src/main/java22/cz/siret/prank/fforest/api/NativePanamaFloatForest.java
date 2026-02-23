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
 * Native C implementation of float-precision forest prediction using Panama FFM (Java 22+).
 * Always uses scalar (non-SIMD) batch prediction. For AVX2, see
 * {@link NativePanamaFloatForestAvx2}.
 *
 * <p>Split points and scores are stored as {@code float} arrays (halving their memory vs double),
 * improving cache utilization during tree traversal. Instance data remains {@code double[]}.
 * The float cast happens at comparison time in C: {@code (float)inst[attr] < splitPoint[node]}.
 *
 * <p>If the native library is not available on the current platform, use
 * {@link #isAvailable()} to check and fall back to {@link IlpDfsFloatForest}.
 */
public class NativePanamaFloatForest implements BinaryForest, Classifier, AutoCloseable {

    static final boolean NATIVE_LOADED;

    // Method handles for native functions (resolved once at class load)
    static final MethodHandle FF_FLOAT_FOREST_CREATE;
    static final MethodHandle FF_FLOAT_FOREST_DESTROY;
    static final MethodHandle FF_FLOAT_PREDICT;
    static final MethodHandle FF_FLOAT_PREDICT_BATCH_SCALAR;
    static final MethodHandle FF_FLOAT_PREDICT_BATCH_AUTO;

    static {
        boolean loaded = false;
        MethodHandle create = null, destroy = null, predict = null,
                     batchScalar = null, batchAuto = null;

        try {
            loaded = NativeLoader.load();
            if (loaded) {
                Linker linker = Linker.nativeLinker();
                SymbolLookup lookup = SymbolLookup.loaderLookup();

                FunctionDescriptor batchDesc = FunctionDescriptor.ofVoid(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS);

                create = linker.downcallHandle(
                        lookup.find("ff_float_forest_create").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS)
                );

                destroy = linker.downcallHandle(
                        lookup.find("ff_float_forest_destroy").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
                );

                predict = linker.downcallHandle(
                        lookup.find("ff_float_predict").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE,
                                ValueLayout.ADDRESS, ValueLayout.ADDRESS)
                );

                batchScalar = linker.downcallHandle(
                        lookup.find("ff_float_predict_batch_scalar_only").orElseThrow(),
                        batchDesc
                );

                batchAuto = linker.downcallHandle(
                        lookup.find("ff_float_predict_batch").orElseThrow(),
                        batchDesc
                );
            }
        } catch (Throwable t) {
            System.err.println("[FasterForest] Failed to initialize native float bindings: " + t.getMessage());
            loaded = false;
        }

        NATIVE_LOADED = loaded;
        FF_FLOAT_FOREST_CREATE = create;
        FF_FLOAT_FOREST_DESTROY = destroy;
        FF_FLOAT_PREDICT = predict;
        FF_FLOAT_PREDICT_BATCH_SCALAR = batchScalar;
        FF_FLOAT_PREDICT_BATCH_AUTO = batchAuto;
    }

//===============================================================================================//

    protected final int numTrees;
    protected final int numAttributes;
    protected final Arena arena;
    protected final MemorySegment forestHandle;
    protected final MethodHandle ffPredictBatch;

    protected NativePanamaFloatForest(int numTrees, int numAttributes, Arena arena,
                                      MemorySegment forestHandle, MethodHandle ffPredictBatch) {
        this.numTrees = numTrees;
        this.numAttributes = numAttributes;
        this.arena = arena;
        this.forestHandle = forestHandle;
        this.ffPredictBatch = ffPredictBatch;
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
     * Build from trained FasterTrees.
     *
     * @throws IllegalStateException if native library is not available
     */
    public static NativePanamaFloatForest fromFasterTrees(int numAttributes, List<FasterTree> trees) {
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
    public static NativePanamaFloatForest fromContiguousDfsForest(ContiguousDfsForest base) {
        if (!NATIVE_LOADED) {
            throw new IllegalStateException("Native library not available");
        }

        Arena arena = Arena.ofShared();
        try {
            MemorySegment handle = createFloatForestHandle(arena, base);
            return new NativePanamaFloatForest(base.numTrees, base.numAttributes, arena, handle,
                    FF_FLOAT_PREDICT_BATCH_SCALAR);
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        } catch (Throwable t) {
            arena.close();
            throw new RuntimeException("Failed to create native float forest", t);
        }
    }

    /**
     * Copy forest arrays to off-heap memory, converting splitPoint and score to float,
     * and create a native float forest handle.
     */
    protected static MemorySegment createFloatForestHandle(Arena arena, ContiguousDfsForest base) throws Throwable {
        MemorySegment treeRootsSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.treeRoots);
        MemorySegment childLeftSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.childLeft);
        MemorySegment childRightSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.childRight);
        MemorySegment attrIndexSeg = arena.allocateFrom(ValueLayout.JAVA_INT, base.attributeIndex);

        // Convert double[] to float[] in Java before copying off-heap
        float[] splitPointFloat = toFloatArray(base.splitPoint);
        float[] scoreFloat = toFloatArray(base.score);

        MemorySegment splitPointSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, splitPointFloat);
        MemorySegment scoreSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, scoreFloat);

        int totalNodes = base.childLeft.length;
        int totalLeaves = base.score.length;

        MemorySegment handle = (MemorySegment) FF_FLOAT_FOREST_CREATE.invokeExact(
                base.numTrees, base.numAttributes,
                totalNodes, totalLeaves,
                treeRootsSeg, childLeftSeg,
                childRightSeg, attrIndexSeg,
                splitPointSeg, scoreSeg
        );

        if (handle.equals(MemorySegment.NULL)) {
            throw new RuntimeException("ff_float_forest_create returned NULL");
        }

        return handle;
    }

    private static float[] toFloatArray(double[] src) {
        float[] dst = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            dst[i] = (float) src[i];
        }
        return dst;
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
            return (double) FF_FLOAT_PREDICT.invokeExact(forestHandle, instSeg);
        } catch (Throwable t) {
            throw new RuntimeException("Native float predict failed", t);
        }
    }

    @Override
    public double[] predictForBatch(double[][] instances) {
        final int n = instances.length;
        if (n == 0) return new double[0];

        try (Arena callArena = Arena.ofConfined()) {
            final long rowBytes = (long) numAttributes * Double.BYTES;

            MemorySegment instanceBuffer = callArena.allocate(
                    (long) n * numAttributes * Double.BYTES, Double.BYTES);
            MemorySegment outputBuffer = callArena.allocate(
                    (long) n * Double.BYTES, Double.BYTES);

            for (int i = 0; i < n; i++) {
                MemorySegment src = MemorySegment.ofArray(instances[i]);
                long copyBytes = (long) instances[i].length * Double.BYTES;
                MemorySegment.copy(src, 0, instanceBuffer, (long) i * rowBytes, copyBytes);
            }

            ffPredictBatch.invokeExact(forestHandle, instanceBuffer, n, outputBuffer);

            double[] result = new double[n];
            MemorySegment dst = MemorySegment.ofArray(result);
            MemorySegment.copy(outputBuffer, 0, dst, 0, (long) n * Double.BYTES);
            return result;
        } catch (Throwable t) {
            throw new RuntimeException("Native float predictForBatch failed", t);
        }
    }

    /**
     * Batch prediction from a contiguous off-heap buffer. Zero-copy path.
     */
    public double[] predictForBatchContiguous(MemorySegment data, int n) {
        if (n == 0) return new double[0];

        try (Arena callArena = Arena.ofConfined()) {
            MemorySegment outputBuffer = callArena.allocate(
                    (long) n * Double.BYTES, Double.BYTES);

            ffPredictBatch.invokeExact(forestHandle, data, n, outputBuffer);

            double[] result = new double[n];
            MemorySegment dst = MemorySegment.ofArray(result);
            MemorySegment.copy(outputBuffer, 0, dst, 0, (long) n * Double.BYTES);
            return result;
        } catch (Throwable t) {
            throw new RuntimeException("Native float predictForBatchContiguous failed", t);
        }
    }

//===============================================================================================//

    @Override
    public void close() {
        try {
            FF_FLOAT_FOREST_DESTROY.invokeExact(forestHandle);
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
