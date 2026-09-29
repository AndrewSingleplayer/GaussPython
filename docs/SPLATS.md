# Gaussian splats for AR on phones

`examples/splat/splat.ha` is a complete Gaussian splatting renderer written in
HA++. It runs on Vulkan (Android, PC), on Metal (iPhone), and its CPU parts
run on every target.

## Why splats are slow on phones, and what this pipeline does about it

| Problem | What the pipeline does |
|---|---|
| **Memory bandwidth.** A trained splat with full spherical harmonics is 59–62 floats in the `.ply`, about 240 bytes; 1M splats is about 240 MB. | `pack_splats` stores 32 bytes per splat: position, RGBA8 colour+opacity, and the 3D covariance as six f16 values precomputed at load time. That is about 7.5× less data read per frame. Real mobile/web viewers use similar 16–32 B formats (PlayCanvas compressed PLY: 16 B; Spark `PackedSplats`: 16 B). |
| **Depth sorting** of every splat, every frame. | A GPU radix sort (4 passes × 8 bits, stable) of 32-bit `(tile << 16) \| depth` keys. There is no CPU sort and no copy back to the CPU. |
| **Overdraw** (many see-through layers per pixel). | A tile rasterizer: each 16×16 tile blends only its own splats, front to back, and **stops per pixel** once it's opaque (transmittance < 0.0001), as in the original 3DGS. |
| Off-screen and tiny splats. | Culled in `preprocess` (near/far planes, zero-area, tile bounds). |

## Frame steps

| Step | Kernel | Work |
|---|---|---|
| 1 | `preprocess` | camera-space position → 2D covariance (EWA) → conic, radius, tile rectangle, 16-bit depth key |
| 2 | `scan_blocks`, `scan_sums`, `scan_add` | prefix sum of "tiles touched" → where each splat writes |
| 3 | `bin` | one key/value pair per (splat, tile) |
| 4 | `radix_hist`, scan, `radix_scatter` ×4 | sort the pairs by tile, then depth |
| 5 | `clear`, `ranges` | start and end of each tile's list |
| 6 | `render` | per pixel: `alpha = min(0.99, opacity · exp(-½ dᵀ Σ⁻¹ d))`, front-to-back blending |

The host needs two GPU submits per frame. After steps 1–2 it reads one number
(the total pair count) to size the sort. The hosts are:
- **Python (PC):** `examples/splat/render.py`, tested.
- **Java (Android):** `examples/splat/android/SplatRenderer.java`, tested on
  a desktop JVM.
- **Swift (iPhone):** `examples/splat/apple/SplatRenderer.swift`, not
  compiled yet.

## Verified on the build machine

The GPU pipeline was run on the lavapipe Vulkan driver and compared with an
independent NumPy implementation (`tests/splat_reference.py`):
- **Packing:** packed splats are bit-identical.
- **Sort:** the sorted keys and the sort order are identical.
- **Image:** every pixel is within 1/255.

The Java renderer produces the same bytes as the Python one.

## AR integration

- **Camera convention:** camera space in `splat.ha` is x right, y down,
  z forward (like 3DGS/OpenCV). ARKit and ARCore view matrices are y up,
  z backward, so negate rows 1 and 2 of their view matrix.
- **Intrinsics:** take fx, fy, cx, cy from the AR framework's camera
  intrinsics, scaled to your render resolution.
- **Compositing:** draw the RGBA image over the camera feed. For proper
  transparency, render with `background = 0` and blend with the stored
  transmittance. Exporting transmittance as alpha is a small change in
  `render`: write `1 - trans` into alpha.

## Next steps for speed on phones (not done yet)

1. **SH degree 1** (view-dependent colour) in a 48-byte format.
2. **Sort every N frames,** or sort only when the camera moves far. Many
   viewers sort at a lower rate.
3. **Sort-free rendering:** Mobile-GS (ICLR 2026) reports 116 FPS on a
   Snapdragon 8 Gen 3 with depth-aware order-independent transparency.
   Qualcomm's weighted-sum rendering is another option.
4. **Level of detail / splat budget:** PlayCanvas suggests about 1M splats on
   mobile.
5. **Subgroup-based ranking** in `radix_scatter`. It currently uses a simple
   O(256) loop per key.

## Sources

- PlayCanvas splat performance and renderers: https://developer.playcanvas.com/user-manual/gaussian-splatting/building/performance/ ,
  https://developer.playcanvas.com/user-manual/gaussian-splatting/rendering-architecture/renderers/
- PlayCanvas compressed splats: https://blog.playcanvas.com/compressing-gaussian-splats/
- Spark (World Labs) PackedSplats: https://sparkjs.dev/docs/packed-splats
- antimatter15/splat (WebGL, 32-byte .splat): https://github.com/antimatter15/splat
- Mobile-GS: https://arxiv.org/abs/2603.11531
- Qualcomm sort-free Weighted Sum Rendering: https://arxiv.org/html/2410.18931v1
- 3D Gaussian Splatting (Kerbl et al. 2023): https://arxiv.org/abs/2308.04079
- Android 16 KB pages: https://developer.android.com/guide/practices/page-sizes
- Vulkan on Android (Vulkan 1.1 on 64-bit Android 10+): https://developer.android.com/games/develop/vulkan/overview
