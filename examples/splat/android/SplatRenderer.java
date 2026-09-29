package com.happ.splat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Renders Gaussian splats on the GPU with the HA++ kernels from splat.ha.
 *
 * Works on Android (Vulkan) and on desktop Java. Uses only the generated class
 * {@link Splat} (JNI methods inside libsplat.so), so no NDK/C++ code is needed.
 *
 * AR use: every frame, take the camera pose from ARCore (Frame.getCamera()),
 * convert its view matrix to x-right / y-down / z-forward (flip rows 1 and 2),
 * and call {@link #render}. Draw the returned RGBA image as a texture over the
 * camera feed, or blend it yourself.
 */
public final class SplatRenderer implements AutoCloseable {
    public static final int TILE = 16;
    static final int RAW_BYTES = 56, PACKED_BYTES = 32, PROJ_BYTES = 44, CAMERA_BYTES = 104;

    private final long gpu;
    private final long kPre, kScanBlocks, kScanSums, kScanAdd, kBin, kHist, kScatter, kClear, kRanges, kRender;
    private final java.util.HashMap<String, long[]> buffers = new java.util.HashMap<>();
    private int n;

    public SplatRenderer() {
        gpu = Splat.gpuCreate();
        if (gpu == 0) throw new IllegalStateException("Vulkan is not available on this device");
        kPre = Splat.preprocessKernel(gpu);
        kScanBlocks = Splat.scan_blocksKernel(gpu);
        kScanSums = Splat.scan_sumsKernel(gpu);
        kScanAdd = Splat.scan_addKernel(gpu);
        kBin = Splat.binKernel(gpu);
        kHist = Splat.radix_histKernel(gpu);
        kScatter = Splat.radix_scatterKernel(gpu);
        kClear = Splat.clearKernel(gpu);
        kRanges = Splat.rangesKernel(gpu);
        kRender = Splat.renderKernel(gpu);
    }

    public String gpuName() { return Splat.gpuName(gpu); }

    /** A GPU buffer of at least `bytes`, reused between frames. */
    private long buf(String name, long bytes) {
        long[] b = buffers.get(name);
        if (b != null && b[1] >= bytes) return b[0];
        if (b != null) Splat.bufferDestroy(b[0]);
        long size = Math.max(bytes, 16);
        long handle = Splat.bufferCreate(gpu, size);
        buffers.put(name, new long[] {handle, size});
        return handle;
    }

    private ByteBuffer data(String name) {
        return Splat.bufferData(buffers.get(name)[0]).order(ByteOrder.nativeOrder());
    }

    /** raw: `count` GaussianRaw records (56 bytes each, see splat.ha), in a direct ByteBuffer. */
    public void upload(ByteBuffer raw, int count) {
        n = count;
        long splats = buf("splats", (long) n * PACKED_BYTES);
        Splat.pack_splats(raw, Splat.bufferData(splats), n);    // HA++ code on the CPU
        buf("projs", (long) n * PROJ_BYTES);
        buf("counts", 4L * n);
        buf("offsets", 4L * n);
        buf("sums", 4L * ((n + 255) / 256));
        buf("camera", CAMERA_BYTES);
    }

    private void scan(String data, int count, String sums) {
        int nb = (count + 255) / 256;
        long d = buffers.get(data)[0], s = buffers.get(sums)[0];
        Splat.scan_blocksRecord(gpu, kScanBlocks, d, s, count, nb, 1, 1);
        Splat.scan_sumsRecord(gpu, kScanSums, s, nb, 1, 1, 1);
        Splat.scan_addRecord(gpu, kScanAdd, d, s, count, nb, 1, 1);
    }

    private long b(String name) { return buffers.get(name)[0]; }

    /**
     * Render one frame. view: world->camera matrix, 16 floats column-major (x right, y down, z forward).
     * Returns width*height RGBA8 pixels (a view of GPU memory, valid until the next render).
     */
    public ByteBuffer render(float[] view, float fx, float fy, float cx, float cy, int width, int height,
                             float near, float far, int backgroundRgba) {
        int tilesX = (width + TILE - 1) / TILE, tilesY = (height + TILE - 1) / TILE;
        ByteBuffer cam = data("camera");
        for (int i = 0; i < 16; i++) cam.putFloat(i * 4, view[i]);
        cam.putFloat(64, fx).putFloat(68, fy).putFloat(72, cx).putFloat(76, cy);
        cam.putFloat(80, width).putFloat(84, height).putFloat(88, near).putFloat(92, far);
        cam.putInt(96, tilesX).putInt(100, tilesY);

        // phase A: project splats, prefix-sum the tiles they touch
        check(Splat.batchBegin(gpu));
        Splat.preprocessRecord(gpu, kPre, b("splats"), b("camera"), b("projs"), b("counts"), b("offsets"), n,
                               (n + 127) / 128, 1, 1);
        scan("offsets", n, "sums");
        check(Splat.batchSubmit(gpu));
        int total = n == 0 ? 0 : data("offsets").getInt((n - 1) * 4) + data("counts").getInt((n - 1) * 4);

        // phase B: bin, radix sort, tile ranges, render
        int ntiles = tilesX * tilesY, nb = Math.max(1, (total + 255) / 256);
        buf("keys0", 4L * total); buf("vals0", 4L * total); buf("keys1", 4L * total); buf("vals1", 4L * total);
        buf("hist", 4L * 256 * nb); buf("hsums", 4L * nb); buf("spans", 8L * ntiles);
        buf("image", 4L * width * height);
        check(Splat.batchBegin(gpu));
        if (total > 0) {
            Splat.binRecord(gpu, kBin, b("projs"), b("offsets"), b("counts"), b("keys0"), b("vals0"), n, tilesX,
                            (n + 127) / 128, 1, 1);
            String[] src = {"keys0", "vals0"}, dst = {"keys1", "vals1"};
            for (int shift = 0; shift < 32; shift += 8) {
                Splat.radix_histRecord(gpu, kHist, b(src[0]), b("hist"), total, shift, nb, nb, 1, 1);
                scan("hist", 256 * nb, "hsums");
                Splat.radix_scatterRecord(gpu, kScatter, b(src[0]), b(src[1]), b(dst[0]), b(dst[1]), b("hist"),
                                          total, shift, nb, nb, 1, 1);
                String[] t = src; src = dst; dst = t;
            }
        }
        Splat.clearRecord(gpu, kClear, b("spans"), 2 * ntiles, (2 * ntiles + 255) / 256, 1, 1);
        if (total > 0) Splat.rangesRecord(gpu, kRanges, b("keys0"), b("spans"), total, (total + 255) / 256, 1, 1);
        Splat.renderRecord(gpu, kRender, b("projs"), b("vals0"), b("spans"), b("image"), width, height,
                           backgroundRgba, tilesX, tilesY, 1);
        check(Splat.batchSubmit(gpu));
        lastPairs = total;
        return data("image");
    }

    /** Tile pairs sorted in the last frame (a measure of work). */
    public int lastPairs;

    private static void check(int rc) {
        if (rc != 0) throw new IllegalStateException("GPU error " + rc);
    }

    @Override
    public void close() {
        for (long[] b : buffers.values()) Splat.bufferDestroy(b[0]);
        buffers.clear();
        for (long k : new long[] {kPre, kScanBlocks, kScanSums, kScanAdd, kBin, kHist, kScatter, kClear, kRanges,
                                  kRender})
            Splat.kernelDestroy(k);
        Splat.gpuDestroy(gpu);
    }
}
