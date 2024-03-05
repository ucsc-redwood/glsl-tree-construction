#pragma once
#include "application.hpp"
#include "core/VulkanTools.h"
#include <iostream>

#define BUFFER_ELEMENTS 131072

class RadixSort : public Application{
    public:
    RadixSort() = default;
    ~RadixSort() { cleanup(); };
	VkResult     create_instance() override;
	void         create_device() override;
	void         create_compute_queue() override;
	void         build_command_pool() override;
	void 		 create_command_buffer(const VkDescriptorSet *descriptor_set, uint32_t logical_block) override;
	void  		 create_storage_buffer(const VkDeviceSize bufferSize, void* data, VkBuffer* device_buffer, VkDeviceMemory* device_memory, VkBuffer* host_buffer, VkDeviceMemory* host_memory) override;
	void 		 build_compute_pipeline() override;
    void         execute();
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


void RadixSort::build_compute_pipeline(){
			printf("build_compute_pipeline\n");
			std::vector<VkDescriptorPoolSize> poolSizes = {
				VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
			};

			VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = 1;
			vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);
			printf("create bindings\n");
			// create binidngs
			VkDescriptorSetLayoutBinding b_sort_layoutBinding{};
			b_sort_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b_sort_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			b_sort_layoutBinding.binding = 0;
			b_sort_layoutBinding.descriptorCount = 1;

			
			VkDescriptorSetLayoutBinding b_alt_layoutBinding{};
			b_alt_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b_alt_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			b_alt_layoutBinding.binding = 1;
			b_alt_layoutBinding.descriptorCount = 1;

			VkDescriptorSetLayoutBinding b_global_hist_layoutBinding{};
			b_global_hist_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b_global_hist_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			b_global_hist_layoutBinding.binding = 2;
			b_global_hist_layoutBinding.descriptorCount = 1;

			VkDescriptorSetLayoutBinding b_index_layoutBinding{};
			b_index_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b_index_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			b_index_layoutBinding.binding = 3;
			b_index_layoutBinding.descriptorCount = 1;

			VkDescriptorSetLayoutBinding b_pass_hist_layoutBinding{};
			b_pass_hist_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			b_pass_hist_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			b_pass_hist_layoutBinding.binding = 4;
			b_pass_hist_layoutBinding.descriptorCount = 1;

			VkDescriptorSetLayoutBinding param_layoutBinding{};
			param_layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			param_layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			param_layoutBinding.binding = 5;
			param_layoutBinding.descriptorCount = 2;


			std::vector<VkDescriptorSetLayoutBinding> histogram_set_layout_bindings = {
				b_sort_layoutBinding, b_global_hist_layoutBinding
			};
			std::vector<VkDescriptorSetLayoutBinding> binning_set_layout_bindings = {
				b_sort_layoutBinding, b_alt_layoutBinding, b_global_hist_layoutBinding, b_index_layoutBinding, b_pass_hist_layoutBinding, param_layoutBinding
			};


			printf("create two descriptor set layouts\n");
			// create two descriptor set layouts for histogram and binning respectively
			VkDescriptorSetLayoutCreateInfo histogram_descriptorLayout{};
			histogram_descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			histogram_descriptorLayout.pBindings = histogram_set_layout_bindings.data();
			histogram_descriptorLayout.bindingCount = static_cast<uint32_t>(histogram_set_layout_bindings.size());
			descriptorLayout[0] = histogram_descriptorLayout;
			vkCreateDescriptorSetLayout(device, &descriptorLayout[0], nullptr, &descriptorSetLayouts[0]);
			
			VkDescriptorSetLayoutCreateInfo binning_descriptorLayout{};
			binning_descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			binning_descriptorLayout.pBindings = binning_set_layout_bindings.data();
			binning_descriptorLayout.bindingCount = static_cast<uint32_t>(binning_set_layout_bindings.size());
			descriptorLayout[1] = binning_descriptorLayout;
			vkCreateDescriptorSetLayout(device, &descriptorLayout[1], nullptr, &descriptorSetLayouts[1]);

			
			printf("create pipeline layout\n");
			// create a pipeline layout with two set layouts
			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
			pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutCreateInfo.setLayoutCount = 2;
			pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;

			vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

			printf("allocate descriptor sets\n");
			// allocate descriptor sets for histogram and binning
			VkDescriptorSetAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.pSetLayouts = descriptorSetLayouts;
			allocInfo.descriptorSetCount = 2;
			vkAllocateDescriptorSets(device, &allocInfo, descriptorSets);
	
			printf("update b sort descriptor sets\n");

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
			// update the histogram descriptor set 
			VkDescriptorBufferInfo b_sort_bufferDescriptor = { radix_sort_buffer.b_sort_buffer, 0, VK_WHOLE_SIZE };
			VkDescriptorBufferInfo g_histogram_bufferDescriptor = { radix_sort_buffer.g_histogram_buffer, 0, VK_WHOLE_SIZE };
			
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[0];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &b_sort_bufferDescriptor;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[0];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &g_histogram_bufferDescriptor;


			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, NULL);



			VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
			pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);
			printf("create pipeline\n");
			// Create pipeline
			VkComputePipelineCreateInfo computePipelineCreateInfo {};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.layout = pipelineLayout;
			computePipelineCreateInfo.flags = 0;
			/*
			// Pass SSBO size via specialization constant
			// could be used to change the constant in glsl such as input size
			struct SpecializationData {
				uint32_t BUFFER_ELEMENT_COUNT = BUFFER_ELEMENTS;
			} specializationData;
			VkSpecializationMapEntry specializationMapEntry{};
			specializationMapEntry.constantID = 0;
			specializationMapEntry.offset = 0;
			specializationMapEntry.size = sizeof(uint32_t);

			VkSpecializationInfo specializationInfo {};
			specializationInfo.mapEntryCount = 1;
			specializationInfo.pMapEntries = &specializationMapEntry;
			specializationInfo.dataSize = sizeof(SpecializationData);
			specializationInfo.pData = &specializationData;
			*/
			const std::string shadersPath = "/home/zheyuan/vulkan-tree-construction/vulkan/radixsort-vulkan-cpp/shaders/compiled_shaders/";

			VkPipelineShaderStageCreateInfo shaderStage = {};
			shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

            shaderStage.module = tools::loadShader((shadersPath + "histogram.spv").c_str(), device);
			shaderStage.pName = "main";
			//shaderStage.pSpecializationInfo = &specializationInfo;
			shaderModule = shaderStage.module;

			assert(shaderStage.module != VK_NULL_HANDLE);
			computePipelineCreateInfo.stage = shaderStage;
			vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline);

			printf("create command buffer\n");
			// Create a command buffer for compute operations
			VkCommandBufferAllocateInfo cmdBufAllocateInfo {};
			cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cmdBufAllocateInfo.commandPool = commandPool;
			cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cmdBufAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer);

			// Fence for compute CB sync
			VkFenceCreateInfo fenceCreateInfo {};
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
			vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
}

void RadixSort::create_storage_buffer(const VkDeviceSize bufferSize, void* data, VkBuffer* device_buffer, VkDeviceMemory* device_memory, VkBuffer* host_buffer, VkDeviceMemory* host_memory){
		printf("create_storage_buffer\n");
		// Copy input data to VRAM using a staging buffer
		{
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

}



void RadixSort::create_command_buffer(const VkDescriptorSet *descriptor_set, uint32_t logical_block){
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
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, descriptor_set, 0, 0);

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
			const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);
			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(commandBuffer, radix_sort_buffer.b_sort_buffer, temp_buffer.b_sort_buffer, 1, &copyRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			bufferBarrier.buffer = temp_buffer.b_sort_buffer;
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
}

void RadixSort::execute(){
			printf("execute\n");
			std::vector<uint32_t> computeOutput(BUFFER_ELEMENTS);
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
			vkMapMemory(device, temp_memory.b_sort_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange{};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = temp_memory.b_sort_memory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);
			
			// Copy to output
			const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);
			memcpy(computeOutput.data(), mapped, bufferSize);
			vkUnmapMemory(device, temp_memory.b_sort_memory);
}

void RadixSort::cleanup(){
		
		vkDestroyBuffer(device, radix_sort_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(device, temp_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(device, temp_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.g_histogram_memory, nullptr);
		vkDestroyBuffer(device, temp_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(device, temp_memory.g_histogram_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_alt_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_alt_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_index_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_index_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_pass_first_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_pass_first_histogram_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_pass_second_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_pass_second_histogram_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_pass_third_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_pass_third_histogram_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.b_pass_fourth_histogram_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.b_pass_fourth_histogram_memory, nullptr);
		vkDestroyBuffer(device, radix_sort_buffer.pass_num_buffer, nullptr);
		vkFreeMemory(device, radix_sort_memory.pass_num_memory, nullptr);
}

void RadixSort::run(){

	std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
	std::vector<uint32_t> g_histogram(1024, 0);

	// Fill input data
	uint32_t n = 0;
	std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

	const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);
	create_instance();
	create_device();
	create_compute_queue();
	build_command_pool();
	create_storage_buffer(bufferSize, computeInput.data(), &radix_sort_buffer.b_sort_buffer, &radix_sort_memory.b_sort_memory, &temp_buffer.b_sort_buffer, &temp_memory.b_sort_memory);
	create_storage_buffer(bufferSize, g_histogram.data(), &radix_sort_buffer.g_histogram_buffer, &radix_sort_memory.g_histogram_memory, &temp_buffer.g_histogram_buffer, &temp_memory.g_histogram_memory);
	build_compute_pipeline();
	create_command_buffer(descriptorSets, 4);
	execute();
}

