/* HA++ GPU runtime (Vulkan compute). See ha_gpu.h.
 *
 * Freestanding on purpose: it needs only clang's own headers plus a handful
 * of C library functions that every OS provides at run time, so `happ build`
 * can compile it for Android without the NDK.
 */
#include "ha_gpu.h"
#include "ha_vk_min.h"

#if defined(_WIN32)
/* HA++ DLLs have no C runtime: use the Win32 heap and our own mem helpers. */
__declspec(dllimport) void *__stdcall LoadLibraryA(const char *);
__declspec(dllimport) void *__stdcall GetProcAddress(void *, const char *);
__declspec(dllimport) void *__stdcall GetProcessHeap(void);
__declspec(dllimport) void *__stdcall HeapAlloc(void *, unsigned long, size_t);
__declspec(dllimport) int __stdcall HeapFree(void *, unsigned long, void *);
static void *ha_malloc(size_t n) { return HeapAlloc(GetProcessHeap(), 0, n); }
static void *ha_calloc(size_t a, size_t b) { return HeapAlloc(GetProcessHeap(), 0x8 /* ZERO_MEMORY */, a * b); }
static void ha_free(void *p) { if (p) HeapFree(GetProcessHeap(), 0, p); }
static void *ha_memcpy(void *d, const void *s, size_t n) {
    unsigned char *dp = (unsigned char *)d;
    const unsigned char *sp = (const unsigned char *)s;
    while (n--) *dp++ = *sp++;
    return d;
}
static void *ha_memset(void *d, int c, size_t n) {
    unsigned char *dp = (unsigned char *)d;
    while (n--) *dp++ = (unsigned char)c;
    return d;
}
#define malloc ha_malloc
#define calloc ha_calloc
#define free ha_free
#define memcpy ha_memcpy
#define memset ha_memset
static void *lib_open(const char *n) { return LoadLibraryA(n); }
static void *lib_sym(void *h, const char *n) { return GetProcAddress(h, n); }
static const char *const VK_LIBS[] = {"vulkan-1.dll", 0};
#else
void *malloc(size_t);
void *calloc(size_t, size_t);
void free(void *);
void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
void *dlopen(const char *, int);
void *dlsym(void *, const char *);
static void *lib_open(const char *n) { return dlopen(n, 2 /* RTLD_NOW */); }
static void *lib_sym(void *h, const char *n) { return dlsym(h, n); }
#if defined(__APPLE__)
static const char *const VK_LIBS[] = {"libvulkan.1.dylib", "libMoltenVK.dylib", 0};
#elif defined(__ANDROID__)
static const char *const VK_LIBS[] = {"libvulkan.so", 0};
#else
static const char *const VK_LIBS[] = {"libvulkan.so.1", "libvulkan.so", 0};
#endif
#endif

#ifndef HA_MAX_SETS
#define HA_MAX_SETS 256       /* descriptor sets per pool; more pools are chained on demand */
#endif
#define MAX_SETS HA_MAX_SETS
#define MAX_POOLS 64
#define MAX_BATCH_KERNELS 64

typedef struct {
    VkResult (*CreateInstance)(const VkInstanceCreateInfo *, const void *, VkInstance *);
    VkResult (*EnumerateInstanceExtensionProperties)(const char *, uint32_t *, VkExtensionProperties *);
    void (*DestroyInstance)(VkInstance, const void *);
    VkResult (*EnumeratePhysicalDevices)(VkInstance, uint32_t *, VkPhysicalDevice *);
    void (*GetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties *);
    void (*GetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t *, VkQueueFamilyProperties *);
    void (*GetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *);
    void (*GetPhysicalDeviceFeatures2)(VkPhysicalDevice, VkPhysicalDeviceFeatures2 *);
    VkResult (*EnumerateDeviceExtensionProperties)(VkPhysicalDevice, const char *, uint32_t *,
                                                   VkExtensionProperties *);
    VkResult (*CreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo *, const void *, VkDevice *);
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

    void (*DestroyDevice)(VkDevice, const void *);
    void (*GetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue *);
    VkResult (*CreateBuffer)(VkDevice, const VkBufferCreateInfo *, const void *, VkBuffer *);
    void (*DestroyBuffer)(VkDevice, VkBuffer, const void *);
    void (*GetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements *);
    VkResult (*AllocateMemory)(VkDevice, const VkMemoryAllocateInfo *, const void *, VkDeviceMemory *);
    void (*FreeMemory)(VkDevice, VkDeviceMemory, const void *);
    VkResult (*BindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
    VkResult (*MapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void **);
    void (*UnmapMemory)(VkDevice, VkDeviceMemory);
    VkResult (*CreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo *, const void *, VkShaderModule *);
    void (*DestroyShaderModule)(VkDevice, VkShaderModule, const void *);
    VkResult (*CreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo *, const void *,
                                          VkDescriptorSetLayout *);
    void (*DestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void *);
    VkResult (*CreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo *, const void *,
                                     VkPipelineLayout *);
    void (*DestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void *);
    VkResult (*CreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo *,
                                       const void *, VkPipeline *);
    void (*DestroyPipeline)(VkDevice, VkPipeline, const void *);
    VkResult (*CreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo *, const void *,
                                     VkDescriptorPool *);
    void (*DestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void *);
    VkResult (*ResetDescriptorPool)(VkDevice, VkDescriptorPool, VkFlags);
    VkResult (*AllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo *, VkDescriptorSet *);
    void (*UpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet *, uint32_t, const void *);
    VkResult (*CreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo *, const void *, VkCommandPool *);
    void (*DestroyCommandPool)(VkDevice, VkCommandPool, const void *);
    VkResult (*AllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo *, VkCommandBuffer *);
    VkResult (*ResetCommandBuffer)(VkCommandBuffer, VkFlags);
    VkResult (*BeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
    VkResult (*EndCommandBuffer)(VkCommandBuffer);
    void (*CmdBindPipeline)(VkCommandBuffer, int32_t, VkPipeline);
    void (*CmdBindDescriptorSets)(VkCommandBuffer, int32_t, VkPipelineLayout, uint32_t, uint32_t,
                                  const VkDescriptorSet *, uint32_t, const uint32_t *);
    void (*CmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void *);
    void (*CmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    void (*CmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier *,
                               uint32_t, const void *, uint32_t, const void *);
    VkResult (*QueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo *, VkFence);
    VkResult (*CreateFence)(VkDevice, const VkFenceCreateInfo *, const void *, VkFence *);
    void (*DestroyFence)(VkDevice, VkFence, const void *);
    VkResult (*WaitForFences)(VkDevice, uint32_t, const VkFence *, VkBool32, uint64_t);
    VkResult (*ResetFences)(VkDevice, uint32_t, const VkFence *);
    VkResult (*DeviceWaitIdle)(VkDevice);
} ha_vk;

struct ha_gpu {
    void *lib;
    ha_vk vk;
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkQueue queue;
    uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mem;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    char name[256];
    int f16;
    int in_batch;
    int dispatches;
    ha_kernel *used[MAX_BATCH_KERNELS];
    int nused;
};

struct ha_buffer {
    ha_gpu *gpu;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *data;
    size_t size;
};

struct ha_kernel {
    ha_gpu *gpu;
    VkShaderModule module;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkDescriptorPool pools[MAX_POOLS];
    uint32_t npools;             /* pools created so far */
    uint32_t cur;                /* pool currently allocated from (reset after each submit) */
    uint32_t nsets;              /* sets taken from pools[cur] */
    uint32_t num_buffers;
    uint32_t push_bytes;
};

static void set_err(char *err, size_t len, const char *msg) {
    size_t i = 0;
    if (!err || !len) return;
    for (; msg[i] && i + 1 < len; i++) err[i] = msg[i];
    err[i] = 0;
}

static int str_eq(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}

static int has_ext(const VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (str_eq(exts[i].extensionName, name)) return 1;
    return 0;
}

#define LOAD_I(field, name) gpu->vk.field = (void *)gipa(gpu->instance, "vk" name)
#define LOAD_D(field, name) gpu->vk.field = (void *)gpu->vk.GetDeviceProcAddr(gpu->device, "vk" name)

static int type_rank(int32_t t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
    default: return 0;
    }
}

HA_GPU_API ha_gpu *ha_gpu_create(char *err, size_t err_len) {
    ha_gpu *gpu = (ha_gpu *)calloc(1, sizeof(ha_gpu));
    if (!gpu) { set_err(err, err_len, "out of memory"); return 0; }
    for (int i = 0; VK_LIBS[i] && !gpu->lib; i++) gpu->lib = lib_open(VK_LIBS[i]);
    if (!gpu->lib) { set_err(err, err_len, "Vulkan library not found"); free(gpu); return 0; }
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)lib_sym(gpu->lib, "vkGetInstanceProcAddr");
    if (!gipa) { set_err(err, err_len, "vkGetInstanceProcAddr missing"); free(gpu); return 0; }

    LOAD_I(CreateInstance, "CreateInstance");
    LOAD_I(EnumerateInstanceExtensionProperties, "EnumerateInstanceExtensionProperties");
    const char *inst_exts[1];
    uint32_t n_inst_exts = 0;
    VkFlags inst_flags = 0;
    {
        uint32_t n = 0;
        gpu->vk.EnumerateInstanceExtensionProperties(0, &n, 0);
        VkExtensionProperties *exts = (VkExtensionProperties *)calloc(n ? n : 1, sizeof(VkExtensionProperties));
        gpu->vk.EnumerateInstanceExtensionProperties(0, &n, exts);
        if (has_ext(exts, n, "VK_KHR_portability_enumeration")) {    /* MoltenVK */
            inst_exts[n_inst_exts++] = "VK_KHR_portability_enumeration";
            inst_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
        free(exts);
    }
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "HA++", 1, "HA++", 1, HA_VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, inst_flags, &app, 0, 0,
                                n_inst_exts, inst_exts};
    if (gpu->vk.CreateInstance(&ici, 0, &gpu->instance) != VK_SUCCESS) {
        set_err(err, err_len, "vkCreateInstance failed (Vulkan 1.1 needed)");
        free(gpu);
        return 0;
    }
    LOAD_I(DestroyInstance, "DestroyInstance");
    LOAD_I(EnumeratePhysicalDevices, "EnumeratePhysicalDevices");
    LOAD_I(GetPhysicalDeviceProperties, "GetPhysicalDeviceProperties");
    LOAD_I(GetPhysicalDeviceQueueFamilyProperties, "GetPhysicalDeviceQueueFamilyProperties");
    LOAD_I(GetPhysicalDeviceMemoryProperties, "GetPhysicalDeviceMemoryProperties");
    LOAD_I(GetPhysicalDeviceFeatures2, "GetPhysicalDeviceFeatures2");
    LOAD_I(EnumerateDeviceExtensionProperties, "EnumerateDeviceExtensionProperties");
    LOAD_I(CreateDevice, "CreateDevice");
    LOAD_I(GetDeviceProcAddr, "GetDeviceProcAddr");

    /* pick the best device that can run compute work */
    uint32_t ndev = 0;
    gpu->vk.EnumeratePhysicalDevices(gpu->instance, &ndev, 0);
    VkPhysicalDevice devs[16];
    if (ndev > 16) ndev = 16;
    gpu->vk.EnumeratePhysicalDevices(gpu->instance, &ndev, devs);
    int best_rank = -1;
    VkPhysicalDeviceProperties *props = (VkPhysicalDeviceProperties *)calloc(1, sizeof(*props));
    for (uint32_t d = 0; d < ndev; d++) {
        uint32_t nq = 0;
        VkQueueFamilyProperties qf[16];
        gpu->vk.GetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, 0);
        if (nq > 16) nq = 16;
        gpu->vk.GetPhysicalDeviceQueueFamilyProperties(devs[d], &nq, qf);
        int fam = -1;
        for (uint32_t q = 0; q < nq && fam < 0; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) fam = (int)q;
        if (fam < 0) continue;
        gpu->vk.GetPhysicalDeviceProperties(devs[d], props);
        int rank = type_rank(props->deviceType);
        if (rank > best_rank) {
            best_rank = rank;
            gpu->phys = devs[d];
            gpu->qfam = (uint32_t)fam;
            memcpy(gpu->name, props->deviceName, sizeof(gpu->name));
            gpu->name[255] = 0;
        }
    }
    free(props);
    if (best_rank < 0) {
        set_err(err, err_len, "no Vulkan device with compute support");
        gpu->vk.DestroyInstance(gpu->instance, 0);
        free(gpu);
        return 0;
    }
    gpu->vk.GetPhysicalDeviceMemoryProperties(gpu->phys, &gpu->mem);

    /* optional features: f16 math + 16-bit buffers */
    uint32_t next = 0;
    gpu->vk.EnumerateDeviceExtensionProperties(gpu->phys, 0, &next, 0);
    VkExtensionProperties *dexts = (VkExtensionProperties *)calloc(next ? next : 1, sizeof(VkExtensionProperties));
    gpu->vk.EnumerateDeviceExtensionProperties(gpu->phys, 0, &next, dexts);
    const char *dev_exts[4];
    uint32_t n_dev_exts = 0;
    if (has_ext(dexts, next, "VK_KHR_portability_subset")) dev_exts[n_dev_exts++] = "VK_KHR_portability_subset";
    VkPhysicalDeviceShaderFloat16Int8Features f16f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                                                      0, 0, 0};
    VkPhysicalDevice16BitStorageFeatures s16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &f16f,
                                                0, 0, 0, 0};
    VkPhysicalDeviceFeatures2 feats;
    memset(&feats, 0, sizeof(feats));
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = &s16;
    int f16_ext = has_ext(dexts, next, "VK_KHR_shader_float16_int8");
    if (gpu->vk.GetPhysicalDeviceFeatures2) {
        gpu->vk.GetPhysicalDeviceFeatures2(gpu->phys, &feats);
        gpu->f16 = f16f.shaderFloat16 && s16.storageBuffer16BitAccess;
    }
    if (gpu->f16 && f16_ext) dev_exts[n_dev_exts++] = "VK_KHR_shader_float16_int8";
    free(dexts);
    /* enable exactly what we use */
    VkPhysicalDeviceShaderFloat16Int8Features en16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                                                      0, (VkBool32)gpu->f16, 0};
    VkPhysicalDevice16BitStorageFeatures ens16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &en16,
                                                  (VkBool32)gpu->f16, 0, 0, 0};
    VkPhysicalDeviceFeatures2 enabled;
    memset(&enabled, 0, sizeof(enabled));
    enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabled.pNext = gpu->f16 ? (void *)&ens16 : 0;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, gpu->qfam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enabled, 0, 1, &qci, 0, 0,
                              n_dev_exts, dev_exts, 0};
    if (gpu->vk.CreateDevice(gpu->phys, &dci, 0, &gpu->device) != VK_SUCCESS) {
        set_err(err, err_len, "vkCreateDevice failed");
        gpu->vk.DestroyInstance(gpu->instance, 0);
        free(gpu);
        return 0;
    }
    LOAD_D(DestroyDevice, "DestroyDevice");
    LOAD_D(GetDeviceQueue, "GetDeviceQueue");
    LOAD_D(CreateBuffer, "CreateBuffer");
    LOAD_D(DestroyBuffer, "DestroyBuffer");
    LOAD_D(GetBufferMemoryRequirements, "GetBufferMemoryRequirements");
    LOAD_D(AllocateMemory, "AllocateMemory");
    LOAD_D(FreeMemory, "FreeMemory");
    LOAD_D(BindBufferMemory, "BindBufferMemory");
    LOAD_D(MapMemory, "MapMemory");
    LOAD_D(UnmapMemory, "UnmapMemory");
    LOAD_D(CreateShaderModule, "CreateShaderModule");
    LOAD_D(DestroyShaderModule, "DestroyShaderModule");
    LOAD_D(CreateDescriptorSetLayout, "CreateDescriptorSetLayout");
    LOAD_D(DestroyDescriptorSetLayout, "DestroyDescriptorSetLayout");
    LOAD_D(CreatePipelineLayout, "CreatePipelineLayout");
    LOAD_D(DestroyPipelineLayout, "DestroyPipelineLayout");
    LOAD_D(CreateComputePipelines, "CreateComputePipelines");
    LOAD_D(DestroyPipeline, "DestroyPipeline");
    LOAD_D(CreateDescriptorPool, "CreateDescriptorPool");
    LOAD_D(DestroyDescriptorPool, "DestroyDescriptorPool");
    LOAD_D(ResetDescriptorPool, "ResetDescriptorPool");
    LOAD_D(AllocateDescriptorSets, "AllocateDescriptorSets");
    LOAD_D(UpdateDescriptorSets, "UpdateDescriptorSets");
    LOAD_D(CreateCommandPool, "CreateCommandPool");
    LOAD_D(DestroyCommandPool, "DestroyCommandPool");
    LOAD_D(AllocateCommandBuffers, "AllocateCommandBuffers");
    LOAD_D(ResetCommandBuffer, "ResetCommandBuffer");
    LOAD_D(BeginCommandBuffer, "BeginCommandBuffer");
    LOAD_D(EndCommandBuffer, "EndCommandBuffer");
    LOAD_D(CmdBindPipeline, "CmdBindPipeline");
    LOAD_D(CmdBindDescriptorSets, "CmdBindDescriptorSets");
    LOAD_D(CmdPushConstants, "CmdPushConstants");
    LOAD_D(CmdDispatch, "CmdDispatch");
    LOAD_D(CmdPipelineBarrier, "CmdPipelineBarrier");
    LOAD_D(QueueSubmit, "QueueSubmit");
    LOAD_D(CreateFence, "CreateFence");
    LOAD_D(DestroyFence, "DestroyFence");
    LOAD_D(WaitForFences, "WaitForFences");
    LOAD_D(ResetFences, "ResetFences");
    LOAD_D(DeviceWaitIdle, "DeviceWaitIdle");

    gpu->vk.GetDeviceQueue(gpu->device, gpu->qfam, 0, &gpu->queue);
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0,
                                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, gpu->qfam};
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
    if (gpu->vk.CreateCommandPool(gpu->device, &cpci, 0, &gpu->cmd_pool) != VK_SUCCESS ||
        gpu->vk.CreateFence(gpu->device, &fci, 0, &gpu->fence) != VK_SUCCESS) {
        set_err(err, err_len, "could not create command pool/fence");
        ha_gpu_destroy(gpu);
        return 0;
    }
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, gpu->cmd_pool,
                                        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (gpu->vk.AllocateCommandBuffers(gpu->device, &cbai, &gpu->cmd) != VK_SUCCESS) {
        set_err(err, err_len, "could not allocate a command buffer");
        ha_gpu_destroy(gpu);
        return 0;
    }
    return gpu;
}

HA_GPU_API void ha_gpu_destroy(ha_gpu *gpu) {
    if (!gpu) return;
    if (gpu->device) {
        gpu->vk.DeviceWaitIdle(gpu->device);
        if (gpu->fence) gpu->vk.DestroyFence(gpu->device, gpu->fence, 0);
        if (gpu->cmd_pool) gpu->vk.DestroyCommandPool(gpu->device, gpu->cmd_pool, 0);
        gpu->vk.DestroyDevice(gpu->device, 0);
    }
    if (gpu->instance) gpu->vk.DestroyInstance(gpu->instance, 0);
    free(gpu);
}

HA_GPU_API const char *ha_gpu_name(ha_gpu *gpu) { return gpu ? gpu->name : ""; }
HA_GPU_API int ha_gpu_has_f16(ha_gpu *gpu) { return gpu ? gpu->f16 : 0; }

static int find_memory(ha_gpu *gpu, uint32_t bits) {
    const VkFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int fallback = -1;
    for (uint32_t i = 0; i < gpu->mem.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkFlags f = gpu->mem.memoryTypes[i].propertyFlags;
        if ((f & want) != want) continue;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) return (int)i;    /* fast CPU reads */
        if (fallback < 0) fallback = (int)i;
    }
    return fallback;
}

HA_GPU_API ha_buffer *ha_buffer_create(ha_gpu *gpu, size_t bytes) {
    if (!gpu) return 0;
    if (bytes == 0) bytes = 4;
    ha_buffer *b = (ha_buffer *)calloc(1, sizeof(ha_buffer));
    if (!b) return 0;
    b->gpu = gpu;
    b->size = bytes;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, (VkDeviceSize)bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, 0};
    if (gpu->vk.CreateBuffer(gpu->device, &bci, 0, &b->buffer) != VK_SUCCESS) { free(b); return 0; }
    VkMemoryRequirements req;
    gpu->vk.GetBufferMemoryRequirements(gpu->device, b->buffer, &req);
    int type = find_memory(gpu, req.memoryTypeBits);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, req.size, (uint32_t)type};
    if (type < 0 || gpu->vk.AllocateMemory(gpu->device, &mai, 0, &b->memory) != VK_SUCCESS ||
        gpu->vk.BindBufferMemory(gpu->device, b->buffer, b->memory, 0) != VK_SUCCESS ||
        gpu->vk.MapMemory(gpu->device, b->memory, 0, HA_VK_WHOLE_SIZE, 0, &b->data) != VK_SUCCESS) {
        ha_buffer_destroy(b);
        return 0;
    }
    memset(b->data, 0, bytes);
    return b;
}

HA_GPU_API void *ha_buffer_data(ha_buffer *b) { return b ? b->data : 0; }
HA_GPU_API size_t ha_buffer_size(ha_buffer *b) { return b ? b->size : 0; }

HA_GPU_API void ha_buffer_destroy(ha_buffer *b) {
    if (!b) return;
    ha_gpu *gpu = b->gpu;
    if (b->data) gpu->vk.UnmapMemory(gpu->device, b->memory);
    if (b->buffer) gpu->vk.DestroyBuffer(gpu->device, b->buffer, 0);
    if (b->memory) gpu->vk.FreeMemory(gpu->device, b->memory, 0);
    free(b);
}

static int add_pool(ha_kernel *k) {
    if (k->npools >= MAX_POOLS) return 0;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_SETS * k->num_buffers};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, MAX_SETS, 1, &ps};
    if (k->gpu->vk.CreateDescriptorPool(k->gpu->device, &dpci, 0, &k->pools[k->npools]) != VK_SUCCESS) return 0;
    k->npools++;
    return 1;
}

/* One descriptor set for a dispatch. When a pool is full the next one is used (created if needed),
   so a batch can hold any number of dispatches of the same kernel (up to MAX_POOLS * MAX_SETS). */
static VkDescriptorSet alloc_set(ha_kernel *k) {
    for (;;) {
        if (k->nsets >= MAX_SETS) {          /* full (counted here: not every driver reports it) */
            k->cur++;
            k->nsets = 0;
        }
        if (k->cur >= k->npools && !add_pool(k)) return 0;
        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, k->pools[k->cur], 1,
                                            &k->set_layout};
        VkDescriptorSet set = 0;
        if (k->gpu->vk.AllocateDescriptorSets(k->gpu->device, &dsai, &set) == VK_SUCCESS) {
            k->nsets++;
            return set;
        }
        k->nsets = MAX_SETS;                 /* out of pool memory anyway: next pool */
    }
}

HA_GPU_API ha_kernel *ha_kernel_create(ha_gpu *gpu, const void *spirv, size_t spirv_bytes, uint32_t num_buffers,
                                       uint32_t push_bytes) {
    if (!gpu || !spirv || spirv_bytes < 20 || (spirv_bytes & 3)) return 0;
    ha_kernel *k = (ha_kernel *)calloc(1, sizeof(ha_kernel));
    if (!k) return 0;
    k->gpu = gpu;
    k->num_buffers = num_buffers;
    k->push_bytes = (push_bytes + 3u) & ~3u;
    /* SPIR-V must be 4-byte aligned; copy it in case the blob isn't */
    uint32_t *code = (uint32_t *)malloc(spirv_bytes);
    if (!code) { free(k); return 0; }
    memcpy(code, spirv, spirv_bytes);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, spirv_bytes, code};
    VkResult r = gpu->vk.CreateShaderModule(gpu->device, &smci, 0, &k->module);
    free(code);
    if (r != VK_SUCCESS) { ha_kernel_destroy(k); return 0; }

    VkDescriptorSetLayoutBinding binds[32];
    if (num_buffers > 32) { ha_kernel_destroy(k); return 0; }
    for (uint32_t i = 0; i < num_buffers; i++) {
        VkDescriptorSetLayoutBinding b = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0};
        binds[i] = b;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0,
                                             num_buffers, binds};
    if (gpu->vk.CreateDescriptorSetLayout(gpu->device, &dslci, 0, &k->set_layout) != VK_SUCCESS) {
        ha_kernel_destroy(k);
        return 0;
    }
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, k->push_bytes};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &k->set_layout,
                                       k->push_bytes ? 1u : 0u, k->push_bytes ? &pcr : 0};
    if (gpu->vk.CreatePipelineLayout(gpu->device, &plci, 0, &k->layout) != VK_SUCCESS) {
        ha_kernel_destroy(k);
        return 0;
    }
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
                                        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0,
                                         VK_SHADER_STAGE_COMPUTE_BIT, k->module, "main", 0},
                                        k->layout, 0, -1};
    if (gpu->vk.CreateComputePipelines(gpu->device, 0, 1, &cpci, 0, &k->pipeline) != VK_SUCCESS) {
        ha_kernel_destroy(k);
        return 0;
    }
    if (num_buffers && !add_pool(k)) {
        ha_kernel_destroy(k);
        return 0;
    }
    return k;
}

HA_GPU_API void ha_kernel_destroy(ha_kernel *k) {
    if (!k) return;
    ha_gpu *gpu = k->gpu;
    VkDevice d = gpu->device;
    for (int i = 0; i < gpu->nused; i++) {        /* destroyed inside an open batch: forget it */
        if (gpu->used[i] == k) gpu->used[i--] = gpu->used[--gpu->nused];
    }
    for (uint32_t i = 0; i < k->npools; i++) gpu->vk.DestroyDescriptorPool(d, k->pools[i], 0);
    if (k->pipeline) gpu->vk.DestroyPipeline(d, k->pipeline, 0);
    if (k->layout) gpu->vk.DestroyPipelineLayout(d, k->layout, 0);
    if (k->set_layout) gpu->vk.DestroyDescriptorSetLayout(d, k->set_layout, 0);
    if (k->module) gpu->vk.DestroyShaderModule(d, k->module, 0);
    free(k);
}

static void barrier(ha_gpu *gpu, VkFlags src_stage, VkFlags src_access, VkFlags dst_stage, VkFlags dst_access) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, src_access, dst_access};
    gpu->vk.CmdPipelineBarrier(gpu->cmd, src_stage, dst_stage, 0, 1, &mb, 0, 0, 0, 0);
}

HA_GPU_API int ha_batch_begin(ha_gpu *gpu) {
    if (!gpu || gpu->in_batch) return -1;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};
    if (gpu->vk.BeginCommandBuffer(gpu->cmd, &bi) != VK_SUCCESS) return -2;
    gpu->in_batch = 1;
    gpu->dispatches = 0;
    gpu->nused = 0;
    return 0;
}

HA_GPU_API int ha_batch_dispatch(ha_gpu *gpu, ha_kernel *k, ha_buffer *const *buffers, const void *push,
                                 uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!gpu || !k || !gpu->in_batch || k->gpu != gpu) return -1;
    for (uint32_t i = 0; i < k->num_buffers; i++)       /* check everything before recording anything */
        if (!buffers || !buffers[i] || buffers[i]->gpu != gpu) return -4;
    /* the kernel's pools are reset after the submit, so it must be on the list before allocating */
    int seen = 0;
    for (int i = 0; i < gpu->nused; i++)
        if (gpu->used[i] == k) seen = 1;
    if (!seen) {
        if (gpu->nused >= MAX_BATCH_KERNELS) return -5;
        gpu->used[gpu->nused++] = k;
    }
    VkDescriptorSet set = 0;
    if (k->num_buffers) {
        set = alloc_set(k);
        if (!set) return -3;
        VkDescriptorBufferInfo infos[32];
        VkWriteDescriptorSet writes[32];
        for (uint32_t i = 0; i < k->num_buffers; i++) {
            VkDescriptorBufferInfo bi = {buffers[i]->buffer, 0, HA_VK_WHOLE_SIZE};
            infos[i] = bi;
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, set, i, 0, 1,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &infos[i], 0};
            writes[i] = w;
        }
        gpu->vk.UpdateDescriptorSets(gpu->device, k->num_buffers, writes, 0, 0);
    }
    if (gpu->dispatches > 0)
        barrier(gpu, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    gpu->vk.CmdBindPipeline(gpu->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    if (set) gpu->vk.CmdBindDescriptorSets(gpu->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &set, 0, 0);
    if (k->push_bytes && push)
        gpu->vk.CmdPushConstants(gpu->cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k->push_bytes, push);
    gpu->vk.CmdDispatch(gpu->cmd, gx, gy, gz);
    gpu->dispatches++;
    return 0;
}

HA_GPU_API int ha_batch_submit(ha_gpu *gpu) {
    if (!gpu || !gpu->in_batch) return -1;
    barrier(gpu, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
    gpu->in_batch = 0;
    int rc = 0;
    if (gpu->vk.EndCommandBuffer(gpu->cmd) != VK_SUCCESS) {
        rc = -2;
    } else {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &gpu->cmd, 0, 0};
        if (gpu->vk.QueueSubmit(gpu->queue, 1, &si, gpu->fence) != VK_SUCCESS) {
            rc = -6;
        } else {
            if (gpu->vk.WaitForFences(gpu->device, 1, &gpu->fence, 1, ~0ULL) != VK_SUCCESS) rc = -7;
            gpu->vk.ResetFences(gpu->device, 1, &gpu->fence);
        }
    }
    /* whatever happened, leave the context ready for the next batch */
    gpu->vk.ResetCommandBuffer(gpu->cmd, 0);
    for (int i = 0; i < gpu->nused; i++) {
        ha_kernel *k = gpu->used[i];
        for (uint32_t j = 0; j < k->npools && j <= k->cur; j++)
            gpu->vk.ResetDescriptorPool(gpu->device, k->pools[j], 0);
        k->cur = 0;
        k->nsets = 0;
    }
    gpu->nused = 0;
    return rc;
}

HA_GPU_API int ha_dispatch(ha_gpu *gpu, ha_kernel *k, ha_buffer *const *buffers, const void *push, uint32_t gx,
                           uint32_t gy, uint32_t gz) {
    int rc = ha_batch_begin(gpu);
    if (rc) return rc;
    rc = ha_batch_dispatch(gpu, k, buffers, push, gx, gy, gz);
    int rc2 = ha_batch_submit(gpu);
    return rc ? rc : rc2;
}
