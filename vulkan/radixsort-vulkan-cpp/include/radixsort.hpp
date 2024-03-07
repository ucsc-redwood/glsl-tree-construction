#pragma once
#include "application.hpp"
#include "core/VulkanTools.h"
#include "vma_usage.hpp"
#include <iostream>

#define BUFFER_ELEMENTS 131072

class RadixSort : public Application{
    public:
    RadixSort() = default;
    ~RadixSort() { cleanup(); };
	VkResult     create_instance();
	void         create_device();
	void         create_compute_queue();
	void         build_command_pool();
	void 		 create_command_buffer(const VkDescriptorSet *descriptor_sets, uint32_t logical_block);
	void  		 create_storage_buffer(const VkDeviceSize bufferSize, void* data, VkBuffer* device_buffer, VkDeviceMemory* device_memory, VkBuffer* host_buffer, VkDeviceMemory* host_memory);
	void 	     create_descriptor_pool(std::vector<VkDescriptorPoolSize> poolSizes, uint32_t maxSets);
	VkDescriptorSetLayoutBinding 		 build_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding, uint32_t descriptorCount);
	void 		 create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings, VkDescriptorSetLayoutCreateInfo *pCreateInfo, VkDescriptorSetLayout *pSetLayout);
	VkPipelineLayoutCreateInfo 		 init_pipeline_layout(uint32_t setLayoutCount,const VkDescriptorSetLayout *pSetLayouts);
	VkPushConstantRange init_push_constant(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size)
	void 	     add_push_constant(VkPipelineLayoutCreateInfo *pipelineLayoutCreateInfo, VkPushConstantRange *push_constant, const uint32_t push_constant_range_count)
	void 		 allocate_descriptor_sets(uint32_t descriptorSetCount, VkDescriptorSetLayout *descriptorSetLayouts, VkDescriptorSet *descriptorSets);
	void 		 allocate_command_buffer(uint32_t commandBufferCount);
	VkWriteDescriptorSet create_descriptor_write(VkDescriptorSet dstSet, uint32_t dstBinding, VkDescriptorType descriptorType, uint32_t descriptorCount, VkDescriptorBufferInfo *pBufferInfo);
	VkPipelineShaderStageCreateInfo 		 load_shader(const std::string shader_name);
	void 		 create_pipeline(VkPipelineShaderStageCreateInfo *shaderStage, VkPipelineLayout *pipelineLayout, VkPipeline *pipeline);
	VkBufferMemoryBarrier 		 create_buffer_barrier(VkBuffer* buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);
	void 		 create_pipeline_barrier(VkBufferMemoryBarrier* bufferBarrier);
	void 		 create_fence();
    std::vector<uint32_t>         execute();
	void 		 cleanup();
	void run();

    private:

    std::string name = "radix_sort";

	//VkDescriptorSetLayout histogram_descriptorSetLayout;
	//VkDescriptorSet histogram_descriptorSet;
	//VkDescriptorSetLayout binning_descriptorSetLayout;
	//VkDescriptorSet binning_descriptorSet;
	VkDescriptorSetLayout descriptorSetLayouts[2] = {VkDescriptorSetLayout{}, VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[2] = {VkDescriptorSet{}, VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[2] = {VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}};
	struct radix_sort_push_constant{
		uint32_t pass_num = 0;
		uint32_t radix_shift = 0;
	}
	struct{
		VkBuffer b_sort_buffer;
		VkBuffer g_histogram_buffer;
		VkBuffer b_alt_buffer;
		VkBuffer b_index_buffer;
		VkBuffer b_pass_first_histogram_buffer;
		VkBuffer b_pass_second_histogram_buffer;
		VkBuffer b_pass_third_histogram_buffer;
		VkBuffer b_pass_fourth_histogram_buffer;
		VkBuffer pass_num_buffer;
	} radix_sort_buffer;

	struct{
		VkBuffer b_sort_buffer;
		VkBuffer g_histogram_buffer;
		VkBuffer b_alt_buffer;
		VkBuffer b_index_buffer;
		VkBuffer b_pass_first_histogram_buffer;
		VkBuffer b_pass_second_histogram_buffer;
		VkBuffer b_pass_third_histogram_buffer;
		VkBuffer b_pass_fourth_histogram_buffer;
		VkBuffer pass_num_buffer;
		
	} temp_buffer;

	struct{
		VkDeviceMemory b_sort_memory;
		VkDeviceMemory g_histogram_memory;
		VkDeviceMemory b_alt_memory;
		VkDeviceMemory b_index_memory;
		VkDeviceMemory b_pass_first_histogram_memory;
		VkDeviceMemory b_pass_second_histogram_memory;
		VkDeviceMemory b_pass_third_histogram_memory;
		VkDeviceMemory b_pass_fourth_histogram_memory;
		VkDeviceMemory pass_num_memory;
	} radix_sort_memory;

	struct{
		VkDeviceMemory b_sort_memory;
		VkDeviceMemory g_histogram_memory;
		VkDeviceMemory b_alt_memory;
		VkDeviceMemory b_index_memory;
		VkDeviceMemory b_pass_first_histogram_memory;
		VkDeviceMemory b_pass_second_histogram_memory;
		VkDeviceMemory b_pass_third_histogram_memory;
		VkDeviceMemory b_pass_fourth_histogram_memory;
		VkDeviceMemory pass_num_memory;
	} temp_memory;

};


VkResult RadixSort::create_instance(){
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = name.c_str();
	appInfo.pEngineName = name.c_str();
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


void RadixSort::create_device(){
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
		physicalDevice = physicalDevices[0];

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		printf("GPU: %s\n", deviceProperties.deviceName);
}


void RadixSort::create_compute_queue(){
		printf("create_compute_queue\n");
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
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
				break;
			}
		}
		// Create logical device
		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		std::vector<const char*> deviceExtensions = {};


		deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);

		// Get a compute queue
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}


void RadixSort::build_command_pool() {
		printf("build_command_pool\n");
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);
}



VkDescriptorSetLayoutBinding RadixSort::build_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding, uint32_t descriptorCount){
	VkDescriptorSetLayoutBinding layoutBinding{};
	layoutBinding.descriptorType = type;
	layoutBinding.stageFlags = stageFlags;
	layoutBinding.binding = binding;
	layoutBinding.descriptorCount = descriptorCount;
	return layoutBinding;

}

void RadixSort::create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings, VkDescriptorSetLayoutCreateInfo *pCreateInfo, VkDescriptorSetLayout *pSetLayout){
	VkDescriptorSetLayoutCreateInfo descriptorLayout{};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pBindings = descriptor_set_layout_bindings.data();
	descriptorLayout.bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size());
	*pCreateInfo = descriptorLayout;
	vkCreateDescriptorSetLayout(device, pCreateInfo, nullptr, pSetLayout);
}

VkPipelineLayoutCreateInfo RadixSort::init_pipeline_layout(uint32_t setLayoutCount,const VkDescriptorSetLayout *pSetLayouts){
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = setLayoutCount;
	pipelineLayoutCreateInfo.pSetLayouts = pSetLayouts;
}

/*
void RadixSort::create_pipeline_layout(uint32_t setLayoutCount,const VkDescriptorSetLayout *pSetLayouts){
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = setLayoutCount;
	pipelineLayoutCreateInfo.pSetLayouts = pSetLayouts;

	vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

}
*/

VkPushConstantRange RadixSort::init_push_constant(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size){
	VkPushConstantRange push_constant{};
	push_constant.stageFlags = stageFlags;
	push_constant.offset = offset;
	push_constant.size = size;
	return push_constant;
}

void RadixSort::add_push_constant(VkPipelineLayoutCreateInfo *pipelineLayoutCreateInfo, VkPushConstantRange *push_constant, const uint32_t push_constant_range_count){
	pipelineLayoutCreateInfo->pushConstantRangeCount = push_constant_range_count;
	pipelineLayoutCreateInfo->pPushConstantRanges = push_constant;
}


void RadixSort::allocate_command_buffer(uint32_t commandBufferCount){
		VkCommandBufferAllocateInfo cmdBufAllocateInfo {};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = commandPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = commandBufferCount;
		vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer);
}

void RadixSort::allocate_descriptor_sets(uint32_t descriptorSetCount, VkDescriptorSetLayout *descriptorSetLayouts, VkDescriptorSet *descriptorSets){
			VkDescriptorSetAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.pSetLayouts = descriptorSetLayouts;
			allocInfo.descriptorSetCount = descriptorSetCount;
			vkAllocateDescriptorSets(device, &allocInfo, descriptorSets);
}


VkWriteDescriptorSet RadixSort::create_descriptor_write(VkDescriptorSet dstSet, uint32_t dstBinding, VkDescriptorType descriptorType, uint32_t descriptorCount, VkDescriptorBufferInfo *pBufferInfo){
	VkWriteDescriptorSet descriptorWrite {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = dstSet;
	descriptorWrite.dstBinding = dstBinding;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = descriptorCount;
	descriptorWrite.pBufferInfo = pBufferInfo;

	return descriptorWrite;
}

VkPipelineShaderStageCreateInfo RadixSort::load_shader(const std::string shader_name){
	const std::string shadersPath = "/home/zheyuan/vulkan-tree-construction/vulkan/radixsort-vulkan-cpp/shaders/compiled_shaders/";

	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

    shaderStage.module = tools::loadShader((shadersPath + shader_name).c_str(), device);
	shaderStage.pName = "main";
	//shaderStage.pSpecializationInfo = &specializationInfo;
	//shaderModule = shaderStage.module;

	return shaderStage;

}


void RadixSort::create_pipeline(VkPipelineShaderStageCreateInfo *shaderStage, VkPipelineLayout *pipelineLayout, VkPipeline *pipeline){
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);
	printf("create pipeline\n");
	// Create pipeline
	VkComputePipelineCreateInfo computePipelineCreateInfo {};
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.layout = &pipelineLayout;
	computePipelineCreateInfo.flags = 0;

	computePipelineCreateInfo.stage = *shaderStage;
	vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, pipeline);

}

void RadixSort::create_descriptor_pool(std::vector<VkDescriptorPoolSize> poolSizes, uint32_t maxSets){
			printf("build_compute_pipeline\n");

			VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = maxSets;
			vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);

}


void RadixSort::create_storage_buffer(const VkDeviceSize bufferSize, void* data, VkBuffer* device_buffer, VkDeviceMemory* device_memory, VkBuffer* host_buffer, VkDeviceMemory* host_memory){
		printf("create_storage_buffer\n");
		// Copy input data to VRAM using a staging buffer
	
			createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				host_buffer,
				host_memory,
				bufferSize,
				data);

			// Flush writes to host visible buffer
			void* mapped;
			vkMapMemory(device, *host_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = *host_memory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkFlushMappedMemoryRanges(device, 1, &mappedRange);
			vkUnmapMemory(device, *host_memory);

			createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				device_buffer,
				device_memory,
				bufferSize);

			// Copy to staging buffer
			VkCommandBufferAllocateInfo cmdBufAllocateInfo {};
			cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cmdBufAllocateInfo.commandPool = commandPool;
			cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cmdBufAllocateInfo.commandBufferCount = 1;

			VkCommandBuffer copyCmd;
			vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd);
			VkCommandBufferBeginInfo cmdBufInfo {};
			cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			vkBeginCommandBuffer(copyCmd, &cmdBufInfo);

			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(copyCmd, *host_buffer, *device_buffer, 1, &copyRegion);
			vkEndCommandBuffer(copyCmd);

			VkSubmitInfo submitInfo {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &copyCmd;
			VkFenceCreateInfo fenceInfo {};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = 0;
			VkFence fence;
			vkCreateFence(device, &fenceInfo, nullptr, &fence);

			// Submit to the queue
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

			vkDestroyFence(device, fence, nullptr);
			vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);
	

}




VkBufferMemoryBarrier RadixSort::create_buffer_barrier(VkBuffer* buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);{
	VkBufferMemoryBarrier bufferBarrier {};
	bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferBarrier.buffer = *buffer;
	bufferBarrier.size = VK_WHOLE_SIZE;
	bufferBarrier.srcAccessMask = srcAccessMask;
	bufferBarrier.dstAccessMask = dstAccessMask;
	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
}

void RadixSort::create_pipeline_barrier(VkBufferMemoryBarrier* bufferBarrier){
	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_HOST_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_FLAGS_NONE,
		0, nullptr,
		1, bufferBarrier,
		0, nullptr);
}

void RadixSort::create_command_buffer(const VkDescriptorSet *descriptor_sets, uint32_t logical_block){
	/*
			printf("create_command_buffer\n");
			VkCommandBufferBeginInfo cmdBufInfo {};
			cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
			
			// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
			VkBufferMemoryBarrier bufferBarrier {};
			bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			bufferBarrier.buffer = radix_sort_buffer.b_sort_buffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			VkBufferMemoryBarrier g_histogram_bufferBarrier {};
			g_histogram_bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			g_histogram_bufferBarrier.buffer = radix_sort_buffer.g_histogram_buffer;
			g_histogram_bufferBarrier.size = VK_WHOLE_SIZE;
			g_histogram_bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			g_histogram_bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			g_histogram_bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			g_histogram_bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

				vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &g_histogram_bufferBarrier,
				0, nullptr);

			

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, descriptor_sets, 0, 0);

			vkCmdDispatch(commandBuffer, logical_block, 1, 1);

			// Barrier to ensure that shader writes are finished before buffer is read back from GPU
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			bufferBarrier.buffer = radix_sort_buffer.b_sort_buffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			g_histogram_bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			g_histogram_bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			g_histogram_bufferBarrier.buffer = radix_sort_buffer.g_histogram_buffer;
			g_histogram_bufferBarrier.size = VK_WHOLE_SIZE;
			g_histogram_bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			g_histogram_bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);


			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &g_histogram_bufferBarrier,
				0, nullptr);

		
			// todo: change the hardcode
			// Read back to host visible buffer
			
			const VkDeviceSize bufferSize = 1024 * sizeof(uint32_t);
			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(commandBuffer, radix_sort_buffer.g_histogram_buffer, temp_buffer.g_histogram_buffer, 1, &copyRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			bufferBarrier.buffer = temp_buffer.g_histogram_buffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			vkEndCommandBuffer(commandBuffer);
			*/
}


void RadixSort::create_fence(){
	VkFenceCreateInfo fenceCreateInfo {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
}

std::vector<uint32_t> RadixSort::execute(){
			// todo: change the harded coded for map
			printf("execute\n");
			std::vector<uint32_t> computeOutput(1024);
			vkResetFences(device, 1, &fence);
			const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &computeSubmitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

			// Make device writes visible to the host
			void *mapped;
			vkMapMemory(device, temp_memory.g_histogram_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange{};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = temp_memory.g_histogram_memory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);
			
			// todo: change the buffer size
			// Copy to output
			const VkDeviceSize bufferSize = 1024 * sizeof(uint32_t);
			memcpy(computeOutput.data(), mapped, bufferSize);
			vkUnmapMemory(device, temp_memory.g_histogram_memory);

			return computeOutput;
}

void RadixSort::cleanup(){
		vkDestroyBuffer(device, radix_sort_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.g_histogram_memory, nullptr);
		vkDestroyBuffer(device, temp_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(device, temp_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(device, temp_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(device, temp_memory.g_histogram_memory, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts[0], nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts[1], nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineCache(device, pipelineCache, nullptr);
		vkDestroyFence(device, fence, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, shaderModule, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		
}

void RadixSort::run(){

	std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
	std::vector<uint32_t> g_histogram(1024, 0);
	VkPipelineLayout pipelineLayout;
	VkPipeline histogram_pipeline;
	VkPipeline binning_pipeline;

	// Fill input data
	uint32_t n = 131072;
	std::generate(computeInput.begin(), computeInput.end(), [&n] { return n--; });

	const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);
	create_instance();
	create_device();
	create_compute_queue();
	build_command_pool();

	create_storage_buffer(bufferSize, computeInput.data(), &radix_sort_buffer.b_sort_buffer, &radix_sort_memory.b_sort_memory, &temp_buffer.b_sort_buffer, &temp_memory.b_sort_memory);
	create_storage_buffer(1024*sizeof(uint32_t), g_histogram.data(), &radix_sort_buffer.g_histogram_buffer, &radix_sort_memory.g_histogram_memory, &temp_buffer.g_histogram_buffer, &temp_memory.g_histogram_memory);
	
	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9},
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
	};

	create_descriptor_pool(poolSizes, 2);

	// create layout binding
	VkDescriptorSetLayoutBinding b_sort_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_alt_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
	VkDescriptorSetLayoutBinding b_global_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding b_index_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);
	VkDescriptorSetLayoutBinding b_pass_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1);
	VkDescriptorSetLayoutBinding param_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5, 2);
	std::vector<VkDescriptorSetLayoutBinding> histogram_set_layout_bindings = {
		b_sort_layoutBinding, b_global_hist_layoutBinding
	};
	std::vector<VkDescriptorSetLayoutBinding> binning_set_layout_bindings = {
		b_sort_layoutBinding, b_alt_layoutBinding, b_global_hist_layoutBinding, b_index_layoutBinding, b_pass_hist_layoutBinding, param_layoutBinding
	};

	// create descriptor set layout
	create_descriptor_set_layout(histogram_set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);
	create_descriptor_set_layout(binning_set_layout_bindings, &descriptorLayout[1], &descriptorSetLayouts[1]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(2, descriptorSetLayouts);
	// add push constant
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(radix_sort_push_constant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(2, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
	VkDescriptorBufferInfo b_sort_bufferDescriptor = { radix_sort_buffer.b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_bufferDescriptor = { radix_sort_buffer.g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_bufferDescriptor);
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {b_sort_descriptor_write, g_histogram_descriptor_write};
	vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline for histogram
	VkPipelineShaderStageCreateInfo histogram_shader_stage = load_shader("histogram.spv");
	create_pipeline(&histogram_shader_stage,&pipelineLayout, &histogram_pipeline);

	//create pipeline for binning
	VkPipelineShaderStageCreateInfo binning_shader_stage = load_shader("new_radix_sort.spv");
	create_pipeline(&binning_shader_stage,&pipelineLayout, &binning_pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(2);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier b_sort_barrier = create_buffer_Barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier g_histogram_barrier = create_buffer_Barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier);
	create_pipeline_barrier(&g_histogram_barrier);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogram_pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, descriptor_sets, 0, 0);
	vkCmdDispatch(commandBuffer, 2, 1, 1);

	b_sort_barrier = create_buffer_Barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	g_histogram_barrier = create_buffer_Barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier);
	create_pipeline_barrier(&g_histogram_barrier);

	vkEndCommandBuffer(commandBuffer);
	//create_command_buffer(descriptorSets, 2);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	auto output = execute();


	VkDescriptorBufferInfo b_sort_bufferDescriptor = { radix_sort_buffer.b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_descriptor_write  = create_descriptor_write(descriptorSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_bufferDescriptor = { radix_sort_buffer.g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_descriptor_write = create_descriptor_write(descriptorSets[1],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_bufferDescriptor);
	VkDescriptorBufferInfo b_alt_bufferDescriptor = { radix_sort_buffer.b_alt_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_alt_descriptor_write  = create_descriptor_write(descriptorSets[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_alt_bufferDescriptor);
	VkDescriptorBufferInfo b_index_bufferDescriptor = { radix_sort_buffer.b_index_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_index_descriptor_write  = create_descriptor_write(descriptorSets[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_index_bufferDescriptor);
	VkDescriptorBufferInfo b_pass_first_histogram_bufferDescriptor = { radix_sort_buffer.b_pass_first_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_pass_first_histogram_descriptor_write  = create_descriptor_write(descriptorSets[1], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_pass_first_histogram_bufferDescriptor);
	VkDescriptorBufferInfo b_pass_second_histogram_bufferDescriptor = { radix_sort_buffer.b_pass_second_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_pass_second_histogram_descriptor_write  = create_descriptor_write(descriptorSets[1], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_pass_second_histogram_bufferDescriptor);

}

