// Gaussian splat renderer for iPhone / Mac, using the Swift package that
// `happ build examples/splat/splat.ha -t ios` generates (apple/Splat).
//
// NOTE: written against the generated API but not compiled here (no Swift or
// Metal toolchain on the Linux build machine). The same kernels run through
// Vulkan in the tested Java/Python hosts; please report anything that fails.
//
// ARKit: each frame, view = frame.camera.viewMatrix(for: orientation) with
// rows 1 and 2 negated (ARKit is y-up/z-back, splat.ha expects y-down/z-forward),
// and fx, fy, cx, cy from frame.camera.intrinsics scaled to your render size.

import Metal
import Splat

public final class SplatRenderer {
    public let gpu: SplatGPU
    private var buffers: [String: MTLBuffer] = [:]
    private var count = 0
    public private(set) var lastPairs = 0

    public init() throws {
        gpu = try SplatGPU()
    }

    private func buf(_ name: String, _ bytes: Int) -> MTLBuffer {
        if let b = buffers[name], b.length >= bytes { return b }
        let b = gpu.makeBuffer(bytes: bytes)!
        buffers[name] = b
        return b
    }

    private func groups(_ n: Int, _ size: Int) -> MTLSize {
        MTLSize(width: max(1, (n + size - 1) / size), height: 1, depth: 1)
    }

    /// raw: `n` GaussianRaw records (see splat.ha / include/splat.h).
    public func upload(raw: UnsafePointer<GaussianRaw>, n: Int) {
        count = n
        let splats = buf("splats", n * MemoryLayout<PackedSplat>.stride)
        pack_splats(UnsafeMutablePointer(mutating: raw), splats.contents().assumingMemoryBound(to: PackedSplat.self),
                    Int64(n))
        _ = buf("projs", n * MemoryLayout<Proj>.stride)
        _ = buf("counts", n * 4)
        _ = buf("offsets", n * 4)
        _ = buf("sums", 4 * ((n + 255) / 256))
        _ = buf("camera", MemoryLayout<Camera>.stride)
    }

    private func scan(_ enc: MTLComputeCommandEncoder, _ data: MTLBuffer, _ n: Int, _ sums: MTLBuffer) throws {
        let nb = (n + 255) / 256
        try gpu.scan_blocks(enc, data: data, sums: sums, n: UInt32(n), groups: groups(n, 256))
        try gpu.scan_sums(enc, sums: sums, nblocks: UInt32(nb), groups: MTLSize(width: 1, height: 1, depth: 1))
        try gpu.scan_add(enc, data: data, sums: sums, n: UInt32(n), groups: groups(n, 256))
    }

    /// view: 16 floats, column-major world->camera (x right, y down, z forward).
    /// Returns a buffer of width*height RGBA8 pixels.
    public func render(view: [Float], fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
                       near: Float = 0.1, far: Float = 100, background: UInt32 = 0xFF000000) throws -> MTLBuffer {
        let tilesX = (width + 15) / 16, tilesY = (height + 15) / 16
        let cam = buffers["camera"]!.contents()
        let f = cam.assumingMemoryBound(to: Float.self)
        for i in 0..<16 { f[i] = view[i] }
        f[16] = fx; f[17] = fy; f[18] = cx; f[19] = cy
        f[20] = Float(width); f[21] = Float(height); f[22] = near; f[23] = far
        let u = cam.assumingMemoryBound(to: UInt32.self)
        u[24] = UInt32(tilesX); u[25] = UInt32(tilesY)
        let n = count
        let splats = buffers["splats"]!, projs = buffers["projs"]!, counts = buffers["counts"]!
        let offsets = buffers["offsets"]!, sums = buffers["sums"]!, camera = buffers["camera"]!

        // phase A: project + prefix sum (one encoder: Metal orders the dispatches and their memory)
        try gpu.run { enc in
            try gpu.preprocess(enc, splats: splats, cams: camera, projs: projs, counts: counts, offsets: offsets,
                               n: UInt32(n), groups: groups(n, 128))
            try scan(enc, offsets, n, sums)
        }
        let off = offsets.contents().assumingMemoryBound(to: UInt32.self)
        let cnt = counts.contents().assumingMemoryBound(to: UInt32.self)
        let total = n == 0 ? 0 : Int(off[n - 1] + cnt[n - 1])
        lastPairs = total

        // phase B: bin, radix sort, tile ranges, render
        let ntiles = tilesX * tilesY, nb = max(1, (total + 255) / 256)
        var src = (buf("keys0", total * 4), buf("vals0", total * 4))
        var dst = (buf("keys1", total * 4), buf("vals1", total * 4))
        let hist = buf("hist", 256 * nb * 4), hsums = buf("hsums", 4 * nb)
        let spans = buf("spans", 8 * ntiles), image = buf("image", 4 * width * height)
        let keys0 = src.0, vals0 = src.1
        try gpu.run { enc in
            if total > 0 {
                try gpu.bin(enc, projs: projs, offsets: offsets, counts: counts, keys: src.0, vals: src.1,
                            n: UInt32(n), tiles_x: UInt32(tilesX), groups: groups(n, 128))
                for shift in stride(from: 0, to: 32, by: 8) {
                    try gpu.radix_hist(enc, keys: src.0, hist: hist, n: UInt32(total), shift: UInt32(shift),
                                       nblocks: UInt32(nb), groups: groups(total, 256))
                    try scan(enc, hist, 256 * nb, hsums)
                    try gpu.radix_scatter(enc, keys: src.0, vals: src.1, keys_out: dst.0, vals_out: dst.1,
                                          hist: hist, n: UInt32(total), shift: UInt32(shift), nblocks: UInt32(nb),
                                          groups: groups(total, 256))
                    swap(&src, &dst)
                }
            }
            try gpu.clear(enc, data: spans, n: UInt32(2 * ntiles), groups: groups(2 * ntiles, 256))
            if total > 0 {
                try gpu.ranges(enc, keys: keys0, spans: spans, n: UInt32(total), groups: groups(total, 256))
            }
            try gpu.render(enc, projs: projs, vals: vals0, spans: spans, image: image, width: UInt32(width),
                           height: UInt32(height), background: background,
                           groups: MTLSize(width: tilesX, height: tilesY, depth: 1))
        }
        return image
    }
}
