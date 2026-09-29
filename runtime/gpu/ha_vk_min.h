/* Minimal Vulkan declarations used by the HA++ GPU runtime.
 *
 * Only what ha_gpu.c needs, copied from the Vulkan 1.1 API (vulkan_core.h,
 * Apache-2.0 / MIT). Having them here means the runtime compiles with nothing
 * but clang: no Vulkan SDK, no Android NDK. tests/test_vk_layout.c checks every
 * struct offset and enum value against the official header.
 * 64-bit targets only (all HA++ targets are 64-bit).
 */
#ifndef HA_VK_MIN_H
#define HA_VK_MIN_H
#include <stddef.h>
#include <stdint.h>

#define HA_VK_HANDLE(name) typedef struct name##_T *name
HA_VK_HANDLE(VkInstance);
HA_VK_HANDLE(VkPhysicalDevice);
HA_VK_HANDLE(VkDevice);
HA_VK_HANDLE(VkQueue);
HA_VK_HANDLE(VkCommandBuffer);
HA_VK_HANDLE(VkDeviceMemory);
HA_VK_HANDLE(VkBuffer);
HA_VK_HANDLE(VkShaderModule);
HA_VK_HANDLE(VkDescriptorSetLayout);
HA_VK_HANDLE(VkPipelineLayout);
HA_VK_HANDLE(VkPipeline);
HA_VK_HANDLE(VkPipelineCache);
HA_VK_HANDLE(VkDescriptorPool);
HA_VK_HANDLE(VkDescriptorSet);
HA_VK_HANDLE(VkCommandPool);
HA_VK_HANDLE(VkFence);
HA_VK_HANDLE(VkSemaphore);

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef int32_t VkResult;
typedef int32_t VkStructureType;

enum {
    VK_SUCCESS = 0,
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES = 1000082000,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES = 1000083000,
};

enum {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
    VK_QUEUE_COMPUTE_BIT = 0x2,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x1,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x2,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20,
    VK_SHARING_MODE_EXCLUSIVE = 0,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x1,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x2,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x4,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 0x8,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_SHADER_STAGE_COMPUTE_BIT = 0x20,
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x2,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x1,
    VK_ACCESS_SHADER_READ_BIT = 0x20,
    VK_ACCESS_SHADER_WRITE_BIT = 0x40,
    VK_ACCESS_HOST_READ_BIT = 0x2000,
    VK_ACCESS_HOST_WRITE_BIT = 0x4000,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800,
    VK_PIPELINE_STAGE_HOST_BIT = 0x4000,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x1,
};

#define HA_VK_API_VERSION_1_1 ((1u << 22) | (1u << 12))
#define HA_VK_WHOLE_SIZE (~0ULL)

typedef struct VkApplicationInfo {
    VkStructureType sType;
    const void *pNext;
    const char *pApplicationName;
    uint32_t applicationVersion;
    const char *pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const VkApplicationInfo *pApplicationInfo;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
} VkInstanceCreateInfo;

/* Only the leading fields are read; the rest (limits etc.) is reserved space. */
typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    int32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    uint64_t reserved_limits_and_sparse[256];
} VkPhysicalDeviceProperties;

typedef struct VkExtent3D {
    uint32_t width, height, depth;
} VkExtent3D;

typedef struct VkQueueFamilyProperties {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkMemoryType {
    VkFlags propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkPhysicalDeviceFeatures {
    VkBool32 flags[55];
} VkPhysicalDeviceFeatures;

typedef struct VkPhysicalDeviceFeatures2 {
    VkStructureType sType;
    void *pNext;
    VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;

typedef struct VkPhysicalDevice16BitStorageFeatures {
    VkStructureType sType;
    void *pNext;
    VkBool32 storageBuffer16BitAccess;
    VkBool32 uniformAndStorageBuffer16BitAccess;
    VkBool32 storagePushConstant16;
    VkBool32 storageInputOutput16;
} VkPhysicalDevice16BitStorageFeatures;

typedef struct VkPhysicalDeviceShaderFloat16Int8Features {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
} VkPhysicalDeviceShaderFloat16Int8Features;

typedef struct VkExtensionProperties {
    char extensionName[256];
    uint32_t specVersion;
} VkExtensionProperties;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float *pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo *pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char *const *ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char *const *ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures *pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkBufferCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    int32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void *pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t *pCode;
} VkShaderModuleCreateInfo;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    int32_t descriptorType;
    uint32_t descriptorCount;
    VkFlags stageFlags;
    const void *pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding *pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPushConstantRange {
    VkFlags stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout *pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange *pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkFlags stage;
    VkShaderModule module;
    const char *pName;
    const void *pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkDescriptorPoolSize {
    int32_t type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize *pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void *pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout *pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void *pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    int32_t descriptorType;
    const void *pImageInfo;
    const VkDescriptorBufferInfo *pBufferInfo;
    const void *pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void *pNext;
    VkCommandPool commandPool;
    int32_t level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    const void *pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkMemoryBarrier {
    VkStructureType sType;
    const void *pNext;
    VkFlags srcAccessMask;
    VkFlags dstAccessMask;
} VkMemoryBarrier;

typedef struct VkSubmitInfo {
    VkStructureType sType;
    const void *pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore *pWaitSemaphores;
    const VkFlags *pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer *pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore *pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkFenceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
} VkFenceCreateInfo;

typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(VkInstance, const char *);
typedef PFN_vkVoidFunction (*PFN_vkGetDeviceProcAddr)(VkDevice, const char *);

#endif
