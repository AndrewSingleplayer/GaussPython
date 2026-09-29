import com.happ.kernels.Kernels;
import java.nio.*;

public class KernelsTest {
    public static void main(String[] args) {
        long gpu = Kernels.gpuCreate();
        if (gpu == 0) throw new RuntimeException("no GPU");
        System.out.println("GPU from Java: " + Kernels.gpuName(gpu) + ", f16=" + Kernels.gpuHasF16(gpu));
        int n = 100000;
        long keys = Kernels.bufferCreate(gpu, 4L * n), counts = Kernels.bufferCreate(gpu, 4 * 256);
        IntBuffer kb = Kernels.bufferData(keys).order(ByteOrder.nativeOrder()).asIntBuffer();
        int[] ref = new int[256];
        java.util.Random r = new java.util.Random(1);
        for (int i = 0; i < n; i++) { int v = r.nextInt(); kb.put(i, v); ref[v >>> 24]++; }
        long k = Kernels.histKernel(gpu);
        int rc = Kernels.histRun(gpu, k, keys, counts, n, (n + 255) / 256, 1, 1);
        IntBuffer cb = Kernels.bufferData(counts).order(ByteOrder.nativeOrder()).asIntBuffer();
        boolean ok = rc == 0;
        for (int i = 0; i < 256; i++) ok &= cb.get(i) == ref[i];
        System.out.println("histRun from Java (GPU): " + (ok ? "correct" : "WRONG rc=" + rc));
        // batch: two more histograms in one submit -> 3x
        Kernels.batchBegin(gpu);
        Kernels.histRecord(gpu, k, keys, counts, n, (n + 255) / 256, 1, 1);
        Kernels.histRecord(gpu, k, keys, counts, n, (n + 255) / 256, 1, 1);
        rc = Kernels.batchSubmit(gpu);
        ok = rc == 0;
        for (int i = 0; i < 256; i++) ok &= cb.get(i) == 3 * ref[i];
        System.out.println("batch of 2 kernels (GPU): " + (ok ? "correct" : "WRONG"));
        // CPU version of a kernel through JNI: ShortArray (f16 bits) in, direct ByteBuffer out
        short[] h = new short[64];
        for (int i = 0; i < 64; i++) h[i] = Float.floatToFloat16(i * 0.5f);
        ByteBuffer out = ByteBuffer.allocateDirect(64 * 8).order(ByteOrder.nativeOrder());
        Kernels.halfs_cpu(h, out, 64, 1, 1, 1);
        ok = true;
        for (int i = 0; i < 64; i++) {
            ok &= Float.float16ToFloat(out.getShort(i * 8)) == i * 1.0f;      // x = 2*a
            ok &= out.getShort(i * 8 + 6) == h[i];                           // w = a
        }
        System.out.println("halfs_cpu from Java: " + (ok ? "correct" : "WRONG"));
        Kernels.kernelDestroy(k); Kernels.bufferDestroy(keys); Kernels.bufferDestroy(counts); Kernels.gpuDestroy(gpu);
    }
}
