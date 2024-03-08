
#pragma once
#include <vulkan/vulkan.hpp>
#include <unordered_map>
#include <iostream>
#include "core/VulkanTools.h"


class Singleton{
	public: 
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;

	Singleton(const Singleton&) = delete;
	Singleton& operator=(const Singleton&) = delete;
	~Singleton(){
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}
	static Singleton& get_singleton(){
		static Singleton singleton;
		return singleton;
	}
	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr);
	protected:
		std::unordered_map<const char *, bool> device_extensions;
		std::unordered_map<const char *, bool> instance_extensions;
		std::vector<std::string> supportedInstanceExtensions;
		std::vector<const char*> enabledDeviceExtensions;
		std::vector<const char*> enabledInstanceExtensions;
		uint32_t api_version = VK_API_VERSION_1_2;
		bool high_priority_graphics_queue{false};
	

	VkResult create_instance(){
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "tree_construction";
	appInfo.pEngineName = "redwood";
	appInfo.apiVersion = api_version;

	std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

	uint32_t extCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
	if (extCount > 0)
	{
		std::vector<VkExtensionProperties> extensions(extCount);
		if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
		{
			for (VkExtensionProperties& extension : extensions)
			{
				supportedInstanceExtensions.push_back(extension.extensionName);
			}
		}
	}

	// Enabled requested instance extensions
	if (enabledInstanceExtensions.size() > 0)
	{
		for (const char * enabledExtension : enabledInstanceExtensions)
		{
			// Output message if requested extension is not available
			if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
			{
				std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
			}
			instanceExtensions.push_back(enabledExtension);
		}
	}

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

	VkInstanceCreateInfo instanceCreateInfo = {};
	if (enableValidationLayers) {
    	instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    	instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
	} else {
    	instanceCreateInfo.enabledLayerCount = 0;
	}
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.pApplicationInfo = &appInfo;

	if (instanceExtensions.size() > 0)
	{
		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	}

    VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);

    return result;
	}

	void create_device(){
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
		physicalDevice = physicalDevices[0];

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		printf("GPU: %s\n", deviceProperties.deviceName);
	}



	Singleton(){
		create_instance();
		create_device();
	}
};


VkResult Singleton::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data){
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