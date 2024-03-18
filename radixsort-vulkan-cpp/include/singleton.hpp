
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
	uint32_t queueFamilyIndex;
	VkQueue queues[4];

    Singleton(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton& operator=(Singleton&&) = delete;
	~Singleton(){
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}
	static Singleton& get_singleton(){
		if (singleton == nullptr){
			singleton =  new Singleton();
		}
		return *singleton;
	};
	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr);
	void createSharedBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size,  void *data = nullptr);
	protected:
		inline static Singleton* singleton;
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

   void create_compute_queue(){
			// Request a single compute queue
		const float defaultQueuePriority(0.0f);
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				queueFamilyIndex = i;
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = i;
				queueCreateInfo.queueCount = 4;
				queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
				break;
			}
		}

		// todo: chane the queue count to 2
		// Create logical device

		// First, query the supported Vulkan version and features
		VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
		physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		VkPhysicalDeviceVulkan12Features vulkan12Features = {};
		vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		vulkan12Features.storageBuffer8BitAccess = VK_TRUE; // Enable 8-bit storage buffers

		// Chain the structures
		physicalDeviceFeatures2.pNext = &vulkan12Features;

		// Pass physicalDeviceFeatures2 to vkGetPhysicalDeviceFeatures2
		vkGetPhysicalDeviceFeatures2(physicalDevice, &physicalDeviceFeatures2);

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		std::vector<const char*> deviceExtensions = {"VK_KHR_8bit_storage"};


		deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		deviceCreateInfo.pNext = &vulkan12Features;
		vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);

		// Get a compute queue
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queues[0]);
		vkGetDeviceQueue(device, queueFamilyIndex, 1, &queues[1]);
		vkGetDeviceQueue(device, queueFamilyIndex, 2, &queues[2]);
		vkGetDeviceQueue(device, queueFamilyIndex, 3, &queues[3]);
	}



	Singleton(){
		create_instance();
		create_device();
		create_compute_queue();
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
		if (vkAllocateMemory(device, &memAlloc, nullptr, memory) != VK_SUCCESS){
			std::cout <<"cannot allocate memory"<<std::endl;
		}
		if (data != nullptr) {
			void *mapped;
			vkMapMemory(device, *memory, 0, size, 0, &mapped);
			memcpy(mapped, data, size);
			vkUnmapMemory(device, *memory);
		}

		vkBindBufferMemory(device, *buffer, *memory, 0);

		return VK_SUCCESS;
};


void Singleton::createSharedBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data){
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

		if(!memTypeFound){
			std::cout <<"cannot find memory type"<<std::endl;
		}else{
			std::cout <<"memory type found"<<std::endl;
		}
		assert(memTypeFound);
		if (vkAllocateMemory(device, &memAlloc, nullptr, memory) != VK_SUCCESS){
			std::cout <<"cannot allocate memory"<<std::endl;
		}else{
			std::cout <<"memory allocated"<<std::endl;
		}
		if (data != nullptr) {
			void *mapped;
			vkMapMemory(device, *memory, 0, size, 0, &mapped);
			memcpy(mapped, data, size);
			vkUnmapMemory(device, *memory);
		}
		vkBindBufferMemory(device, *buffer, *memory, 0);
};

/*
Singleton& Singleton::get_singleton(){
	if (singleton == nullptr){
		singleton =  new Singleton();
	}
	return *singleton;
}
*/