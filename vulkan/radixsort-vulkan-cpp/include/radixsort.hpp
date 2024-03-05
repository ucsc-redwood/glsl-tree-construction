#pragma once
#include "application.hpp"
#include <iostream>

class RadixSort : public Application{
    public:
    RadixSort();
    ~RadixSort();
	VkResult     create_instance() override;
	void         create_device() override;
	void         create_compute_queue() override;
	void         build_command_pool() override;
	void 		 create_command_buffer() override;
	void  		 create_storage_buffer() override;
	void 		 build_compute_pipeline() override;
    void         run();

    private:
    std::string name = "radix_sort";
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

	VkInstanceCreateInfo instanceCreateInfo = {};
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
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);
}


void RadixSort::build_compute_pipeline(){
			std::vector<VkDescriptorPoolSize> poolSizes = {
				VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
			};

			VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = 1;
			vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);

			VkDescriptorSetLayoutBinding layoutBinding{};
			layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			layoutBinding.binding = 0;
			layoutBinding.descriptorCount = 1;
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
				layoutBinding
			};

			VkDescriptorSetLayoutCreateInfo descriptorLayout{};
			descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			descriptorLayout.pBindings = setLayoutBindings.data();
			descriptorLayout.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
			vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout);

			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
			pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

			vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

			VkDescriptorSetAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.pSetLayouts = &descriptorSetLayout;
			allocInfo.descriptorSetCount = 1;
			vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

			VkDescriptorBufferInfo bufferDescriptor = { deviceBuffer, 0, VK_WHOLE_SIZE };
			VkWriteDescriptorSet computeWriteDescriptorSet{};
			computeWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computeWriteDescriptorSet.dstSet = descriptorSet;
			computeWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			computeWriteDescriptorSet.dstBinding = 0;
			computeWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSet.pBufferInfo = &bufferDescriptor;
			std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets= {computeWriteDescriptorSet};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

			VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
			pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);

			// Create pipeline
			VkComputePipelineCreateInfo computePipelineCreateInfo {};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.layout = pipelineLayout;
			computePipelineCreateInfo.flags = 0;

			// Pass SSBO size via specialization constant
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

			std::string shaderDir = "glsl";
			if (commandLineParser.isSet("shaders")) {
				shaderDir = commandLineParser.getValueAsString("shaders", "glsl");
			}
			const std::string shadersPath = getShaderBasePath() + shaderDir + "/computeheadless/";

			VkPipelineShaderStageCreateInfo shaderStage = {};
			shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

            shaderStage.module = vks::tools::loadShader((shadersPath + "headless.comp.spv").c_str(), device);
			shaderStage.pName = "main";
			shaderStage.pSpecializationInfo = &specializationInfo;
			shaderModule = shaderStage.module;

			assert(shaderStage.module != VK_NULL_HANDLE);
			computePipelineCreateInfo.stage = shaderStage;
			vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline);

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


void RadixSort::create_command_buffer(){
			VkCommandBufferBeginInfo cmdBufInfo {};
			cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

			vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);

			// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
			VkBufferMemoryBarrier bufferBarrier {};
			bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			bufferBarrier.buffer = deviceBuffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

			vkCmdDispatch(commandBuffer, BUFFER_ELEMENTS, 1, 1);

			// Barrier to ensure that shader writes are finished before buffer is read back from GPU
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			bufferBarrier.buffer = deviceBuffer;
			bufferBarrier.size = VK_WHOLE_SIZE;
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);

			// Read back to host visible buffer
			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			bufferBarrier.buffer = hostBuffer;
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

void RadixSort::create_storage_buffer(){
		std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
		std::vector<uint32_t> computeOutput(BUFFER_ELEMENTS);

		// Fill input data
		uint32_t n = 0;
		std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

		const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);

		VkBuffer deviceBuffer, hostBuffer;
		VkDeviceMemory deviceMemory, hostMemory;

		// Copy input data to VRAM using a staging buffer
		{
			createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				&hostBuffer,
				&hostMemory,
				bufferSize,
				computeInput.data());

			// Flush writes to host visible buffer
			void* mapped;
			vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
			mappedRange.memory = hostMemory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkFlushMappedMemoryRanges(device, 1, &mappedRange);
			vkUnmapMemory(device, hostMemory);

			createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				&deviceBuffer,
				&deviceMemory,
				bufferSize);

			// Copy to staging buffer
			VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
			VkCommandBuffer copyCmd;
			vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd);
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			vkBeginCommandBuffer(copyCmd, &cmdBufInfo);

			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(copyCmd, hostBuffer, deviceBuffer, 1, &copyRegion);
			vkEndCommandBuffer(copyCmd);

			VkSubmitInfo submitInfo = vks::initializers::submitInfo();
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &copyCmd;
			VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
			VkFence fence;
			vkCreateFence(device, &fenceInfo, nullptr, &fence);

			// Submit to the queue
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

			vkDestroyFence(device, fence, nullptr);
			vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);
		}
}

void RadixSort::execute(){
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
			vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange{};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = hostMemory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);

			// Copy to output
			memcpy(computeOutput.data(), mapped, bufferSize);
			vkUnmapMemory(device, hostMemory);
}

void RadixSort::cleanup(){
		vkDestroyBuffer(device, deviceBuffer, nullptr);
		vkFreeMemory(device, deviceMemory, nullptr);
		vkDestroyBuffer(device, hostBuffer, nullptr);
		vkFreeMemory(device, hostMemory, nullptr);
}


void RadixSort::run(){
}
void RadixSort::build_command_buffers(){
    VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

}

