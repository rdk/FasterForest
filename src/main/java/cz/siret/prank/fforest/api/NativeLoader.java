package cz.siret.prank.fforest.api;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Loads the native fasterforest library from the JAR or a configured path.
 *
 * <p>Platform detection extracts the matching native library from
 * {@code /native/{os}-{arch}/fasterforest.{dll|so|dylib}} in the classpath,
 * copies it to a temp directory, and loads it via {@link System#load(String)}.
 *
 * <p>Set system property {@code fasterforest.native.path} to override with
 * a direct path to the shared library (useful for development).
 */
class NativeLoader {

    private static volatile boolean loaded = false;
    private static volatile boolean available = false;

    /**
     * Attempt to load the native library. Returns true if successful.
     * Safe to call multiple times — only the first call does work.
     */
    static boolean load() {
        if (loaded) return available;

        synchronized (NativeLoader.class) {
            if (loaded) return available;
            loaded = true;

            try {
                available = doLoad();
            } catch (Throwable t) {
                System.err.println("[FasterForest] Native library not available: " + t.getMessage());
                available = false;
            }
        }
        return available;
    }

    static boolean isAvailable() {
        return available;
    }

    private static boolean doLoad() throws IOException {
        // 1. Check for explicit path override
        String override = System.getProperty("fasterforest.native.path");
        if (override != null && !override.isEmpty()) {
            System.load(override);
            return true;
        }

        // 2. Try loading from classpath resources
        String os = detectOs();
        String arch = detectArch();
        if (os == null || arch == null) return false;

        String libName = libraryName(os);
        String resourcePath = "/native/" + os + "-" + arch + "/" + libName;

        try (InputStream in = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) return false;

            // Extract to temp directory
            Path tempDir = Files.createTempDirectory("fasterforest-native-");
            Path tempLib = tempDir.resolve(libName);
            Files.copy(in, tempLib, StandardCopyOption.REPLACE_EXISTING);
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();

            System.load(tempLib.toAbsolutePath().toString());
            return true;
        }
    }

    private static String detectOs() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("linux")) return "linux";
        if (os.contains("mac") || os.contains("darwin")) return "osx";
        if (os.contains("windows")) return "windows";
        return null;
    }

    private static String detectArch() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.equals("amd64") || arch.equals("x86_64")) return "x86_64";
        if (arch.equals("aarch64") || arch.equals("arm64")) return "aarch64";
        return null;
    }

    private static String libraryName(String os) {
        switch (os) {
            case "windows": return "fasterforest.dll";
            case "osx":     return "fasterforest.dylib";
            default:        return "fasterforest.so";
        }
    }
}
