import com.happ.splat.SplatRenderer;
import java.nio.*;
import java.nio.file.*;

/** Renders raw.bin (GaussianRaw records) with SplatRenderer; writes out.bin (RGBA8). */
public class SplatDemo {
    public static void main(String[] args) throws Exception {
        Path dir = Paths.get(args[0]);
        byte[] raw = Files.readAllBytes(dir.resolve("raw.bin"));
        ByteBuffer rb = ByteBuffer.allocateDirect(raw.length).order(ByteOrder.nativeOrder());
        rb.put(raw).flip();
        ByteBuffer cam = ByteBuffer.wrap(Files.readAllBytes(dir.resolve("cam.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        float[] view = new float[16];
        for (int i = 0; i < 16; i++) view[i] = cam.getFloat(i * 4);
        int w = Integer.parseInt(args[1]), h = Integer.parseInt(args[2]);
        try (SplatRenderer r = new SplatRenderer()) {
            r.upload(rb, raw.length / 56);
            ByteBuffer img = r.render(view, cam.getFloat(64), cam.getFloat(68), cam.getFloat(72), cam.getFloat(76),
                                      w, h, cam.getFloat(88), cam.getFloat(92), 0xFF160A08);
            byte[] out = new byte[w * h * 4];
            img.get(0, out);
            Files.write(dir.resolve("out.bin"), out);
            System.out.println("rendered " + w + "x" + h + " on " + r.gpuName() + ", pairs=" + r.lastPairs);
        }
    }
}
