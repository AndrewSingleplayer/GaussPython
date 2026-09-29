/* HA++ GPU runtime (Vulkan): runs HA++ kernels on Android, Windows and Linux GPUs.
 *
 * Vulkan is loaded at run time (dlopen / LoadLibrary), so apps don't link
 * against it and the file compiles with plain clang (no NDK, no Vulkan SDK).
 * On iPhone/Mac use the Metal source instead (see HAMetal.swift).
 *
 *   ha_gpu *gpu = ha_gpu_create(err, sizeof err);
 *   ha_buffer *b = ha_buffer_create(gpu, n * sizeof(float));
 *   float *p = ha_buffer_data(b);                  // fill it
 *   size_t size; const void *spv = mylib_spirv_mykernel(&size);
 *   ha_kernel *k = ha_kernel_create(gpu, spv, size, 1, sizeof(ha_params_mykernel));
 *   ha_dispatch(gpu, k, &b, &params, (n + 63) / 64, 1, 1);   // waits for the GPU
 *
 * Buffers are host-visible and coherent: the CPU reads/writes them directly,
 * which is the fast path on phones (CPU and GPU share memory).
 */
#ifndef HA_GPU_H
#define HA_GPU_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define HA_GPU_API __declspec(dllexport)
#else
#define HA_GPU_API __attribute__((visibility("default")))
#endif

typedef struct ha_gpu ha_gpu;
typedef struct ha_buffer ha_buffer;
typedef struct ha_kernel ha_kernel;

/* Create a GPU context. Returns NULL (and a message in err) if Vulkan is missing. */
HA_GPU_API ha_gpu *ha_gpu_create(char *err, size_t err_len);
HA_GPU_API void ha_gpu_destroy(ha_gpu *gpu);
HA_GPU_API const char *ha_gpu_name(ha_gpu *gpu);
/* 1 if the device supports f16 arithmetic and 16-bit buffers (hvec/f16 kernels). */
HA_GPU_API int ha_gpu_has_f16(ha_gpu *gpu);

HA_GPU_API ha_buffer *ha_buffer_create(ha_gpu *gpu, size_t bytes);
HA_GPU_API void *ha_buffer_data(ha_buffer *buf);
HA_GPU_API size_t ha_buffer_size(ha_buffer *buf);
HA_GPU_API void ha_buffer_destroy(ha_buffer *buf);

/* num_buffers / push_bytes come from the generated header (HA_KERNEL_<name>_BUFFERS, sizeof params). */
HA_GPU_API ha_kernel *ha_kernel_create(ha_gpu *gpu, const void *spirv, size_t spirv_bytes,
                                       uint32_t num_buffers, uint32_t push_bytes);
HA_GPU_API void ha_kernel_destroy(ha_kernel *k);

/* Run one kernel and wait for it. Returns 0 on success. */
HA_GPU_API int ha_dispatch(ha_gpu *gpu, ha_kernel *k, ha_buffer *const *buffers, const void *push,
                           uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

/* Record several dispatches (e.g. the passes of a GPU sort) and run them in one submit.
   Each dispatch sees the results of the previous ones. */
HA_GPU_API int ha_batch_begin(ha_gpu *gpu);
HA_GPU_API int ha_batch_dispatch(ha_gpu *gpu, ha_kernel *k, ha_buffer *const *buffers, const void *push,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
HA_GPU_API int ha_batch_submit(ha_gpu *gpu);

#ifdef __cplusplus
}
#endif
#endif
