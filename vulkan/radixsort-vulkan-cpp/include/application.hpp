
#pragma once
#include <vulkan/vulkan.hpp>
#include <unordered_map>
#include <vk_mem_alloc.h>

class Application{
	public: 
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	uint32_t queueFamilyIndex;
	VkPipelineCache pipelineCache;
	VkQueue queue;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence fence;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule shaderModule;

	Application() = default;
	virtual ~Application();

	virtual void create_device();

	virtual VkResult create_instance();
	virtual void create_compute_queue();

	virtual void build_command_pool();
	virtual void create_command_buffer();
	virtual void create_storage_buffer();
	virtual void build_compute_pipeline();

	virtual const std::unordered_map<const char *, bool> get_instance_extensions();

	virtual const std::unordered_map<const char *, bool> get_device_extensions();

	virtual void add_device_extension(const char *extension, bool optional = false);

	virtual void add_instance_extension(const char *extension, bool optional = false);

	virtual void set_api_version(uint32_t requested_api_version);

	virtual void request_gpu_features(VULKAN_HPP_NAMESPACE::PhysicalDevice &gpu);
	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr);
	protected:
		std::unordered_map<const char *, bool> device_extensions;
		std::unordered_map<const char *, bool> instance_extensions;
		std::vector<std::string> supportedInstanceExtensions;
		std::vector<const char*> enabledDeviceExtensions;
		std::vector<const char*> enabledInstanceExtensions;
		uint32_t api_version = VK_API_VERSION_1_0;
		bool high_priority_graphics_queue{false};
};


VkResult Application::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data){
		// Create the buffer handle
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usageFlags;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer);

		// Create the memory backing up the buffer handle
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
		VkMemoryRequirements memReqs;
		VkMemoryAllocateInfo memAlloc {};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		vkGetBufferMemoryRequirements(device, *buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Find a memory type index that fits the properties of the buffer
		bool memTypeFound = false;
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
			if ((memReqs.memoryTypeBits & 1) == 1) {
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
					memAlloc.memoryTypeIndex = i;
					memTypeFound = true;
					break;
				}
			}
			memReqs.memoryTypeBits >>= 1;
		}
		assert(memTypeFound);
		vkAllocateMemory(device, &memAlloc, nullptr, memory);

		if (data != nullptr) {
			void *mapped;
			vkMapMemory(device, *memory, 0, size, 0, &mapped);
			memcpy(mapped, data, size);
			vkUnmapMemory(device, *memory);
		}

		vkBindBufferMemory(device, *buffer, *memory, 0);

		return VK_SUCCESS;
}