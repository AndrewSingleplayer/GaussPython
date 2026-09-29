# GPU kernels and the runtime

## One source, three GPU outputs

`happ build` turns every `kernel` into:

| Output | Used on | Check done on the build machine |
|---|---|---|
| GLSL → **SPIR-V** (`gpu/<lib>_<kernel>.spv`, embedded as `<lib>_spirv_<kernel>()`) | Vulkan: Android, Windows, Linux | `glslangValidator`, `spirv-val`, and runs on the lavapipe driver in the tests |
| **Metal** source (`gpu/<lib>.metal`, embedded as `<lib>_metal_source()`) | iPhone, Mac | compiled and run through a C++ stand-in for `metal_stdlib` (tests); not Apple's compiler |
| **CPU** version `<kernel>_cpu(...)` | everywhere | native code, tested on x86-64 and ARM64 |

## Buffer layout is identical everywhere

GPU languages normally pad `vec3` to 16 bytes, while C uses 12. HA++ instead
stores vectors packed in buffers on every platform:
- **GLSL:** `struct ha_p_vec3 { float v[3]; }`, with only scalars in std430.
- **Metal:** `packed_float3`.

So the bytes your app writes, from C, Swift, Kotlin `ByteBuffer` or NumPy, are
the bytes the kernel reads. The C header asserts every struct size.

## Launching kernels (Vulkan runtime)

`runtime/gpu/ha_gpu.c` is linked into Android/Windows/Linux libraries. It:
- loads Vulkan when the app starts, so it isn't linked against it;
- picks the best GPU;
- enables f16 when the device supports it.

C API (`include/ha_gpu.h`):

```c
ha_gpu *gpu = ha_gpu_create(err, sizeof err);
ha_buffer *buf = ha_buffer_create(gpu, bytes);        // CPU-visible (phones share CPU/GPU memory)
float *p = ha_buffer_data(buf);
size_t size; const void *spv = my_spirv_scale(&size);
ha_kernel *k = ha_kernel_create(gpu, spv, size, HA_SCALE_BUFFERS, sizeof(ha_params_scale));
ha_params_scale params = { .n = n, .k = 2.0f };
ha_dispatch(gpu, k, &buf, &params, (n + HA_SCALE_WORKGROUP_X - 1) / HA_SCALE_WORKGROUP_X, 1, 1);

ha_batch_begin(gpu);                                   // many kernels, one submit, ordered
ha_batch_dispatch(gpu, k1, ...); ha_batch_dispatch(gpu, k2, ...);
ha_batch_submit(gpu);                                  // waits
```

The generated Kotlin/Java class has typed wrappers such as
`scaleKernel(gpu)`, `scaleRun(gpu, k, buf, n, 2f, gx, gy, gz)` and
`scaleRecord(...)`. See [PLATFORMS.md](PLATFORMS.md).

## Launching kernels (Metal, Swift)

The generated Swift package has a `<Name>GPU` class. It compiles the embedded
Metal source once, caches the pipelines, and has one typed method per kernel:

```swift
let gpu = try MyGPU()
let buf = gpu.makeBuffer(bytes: n * 4)!
try gpu.scale(data: buf, n: UInt32(n), k: 2, groups: MTLSize(width: (n + 63) / 64, height: 1, depth: 1))
try gpu.run { enc in                       // several kernels in one command buffer
    try gpu.a(enc, ...); try gpu.b(enc, ...)
}
```

## Speed tips for phone GPUs

- **Use `f16`/`hvec`** for data that doesn't need full precision. It halves
  memory traffic, which is what limits phones. Check `gpuHasF16`.
- **Keep work on the GPU.** Chain kernels in one batch and read results back
  only when you must. Each read-back makes the CPU wait for the GPU.
- **Workgroups of 64–256 threads.** Use `shared` memory for data that
  neighbouring threads reuse (see `matmul` in `examples/ai/nn.ha`).
- **Pack data:** see `pack_half2`, `pack_unorm4` and the 32-byte `PackedSplat`.
