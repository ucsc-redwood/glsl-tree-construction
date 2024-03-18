#pragma once
#include "singleton.hpp"

class ApplicationBase{
    public:
        virtual ~ApplicationBase() {
            vkDestroyPipelineCache(singleton.device, pipelineCache, nullptr);
            vkDestroyCommandPool(singleton.device, commandPool, nullptr);
            vkDestroyFence(singleton.device, fence, nullptr);
            vkDestroyDescriptorPool(singleton.device, descriptorPool, nullptr);
            vkDestroyPipelineLayout(singleton.device, pipelineLayout, nullptr);
        }


    protected:
        Singleton &singleton;
        VkCommandPool commandPool;
        VkPipelineCache pipelineCache;
       // uint32_t queueFamilyIndex;
	   // VkQueue queue;
        VkCommandBuffer commandBuffer;
	    VkFence fence;
	    VkDescriptorPool descriptorPool;
	    VkPipelineLayout pipelineLayout;
		
        ApplicationBase(): singleton(Singleton::get_singleton()) {
	        //create_compute_queue();
	        build_command_pool();
        }



    void build_command_pool() {
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = singleton.queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		vkCreateCommandPool(singleton.device, &cmdPoolInfo, nullptr, &commandPool);
	}



VkDescriptorSetLayoutBinding build_layout_binding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding, uint32_t descriptorCount){
	VkDescriptorSetLayoutBinding layoutBinding{};
	layoutBinding.descriptorType = type;
	layoutBinding.stageFlags = stageFlags;
	layoutBinding.binding = binding;
	layoutBinding.descriptorCount = descriptorCount;
	return layoutBinding;

}

void create_descriptor_set_layout(std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings, VkDescriptorSetLayoutCreateInfo *pCreateInfo, VkDescriptorSetLayout *pSetLayout){
	VkDescriptorSetLayoutCreateInfo descriptorLayout{};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pBindings = descriptor_set_layout_bindings.data();
	descriptorLayout.bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size());
	*pCreateInfo = descriptorLayout;
	vkCreateDescriptorSetLayout(singleton.device, pCreateInfo, nullptr, pSetLayout);
}

VkPipelineLayoutCreateInfo init_pipeline_layout(uint32_t setLayoutCount,const VkDescriptorSetLayout *pSetLayouts){
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = setLayoutCount;
	pipelineLayoutCreateInfo.pSetLayouts = pSetLayouts;
	return pipelineLayoutCreateInfo;
}


VkPushConstantRange init_push_constant(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size){
	VkPushConstantRange push_constant{};
	push_constant.stageFlags = stageFlags;
	push_constant.offset = offset;
	push_constant.size = size;
	return push_constant;
}

void add_push_constant(VkPipelineLayoutCreateInfo *pipelineLayoutCreateInfo, VkPushConstantRange *push_constant, const uint32_t push_constant_range_count){
	pipelineLayoutCreateInfo->pushConstantRangeCount = push_constant_range_count;
	pipelineLayoutCreateInfo->pPushConstantRanges = push_constant;
}


void allocate_command_buffer(uint32_t commandBufferCount){
		VkCommandBufferAllocateInfo cmdBufAllocateInfo {};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = commandPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = commandBufferCount;
		vkAllocateCommandBuffers(singleton.device, &cmdBufAllocateInfo, &commandBuffer);
}

void allocate_descriptor_sets(uint32_t descriptorSetCount, VkDescriptorSetLayout *descriptorSetLayouts, VkDescriptorSet *descriptorSets){
			VkDescriptorSetAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = descriptorPool;
			allocInfo.pSetLayouts = descriptorSetLayouts;
			allocInfo.descriptorSetCount = descriptorSetCount;
			vkAllocateDescriptorSets(singleton.device, &allocInfo, descriptorSets);
}


VkWriteDescriptorSet create_descriptor_write(VkDescriptorSet dstSet, uint32_t dstBinding, VkDescriptorType descriptorType, uint32_t descriptorCount, VkDescriptorBufferInfo *pBufferInfo){
	VkWriteDescriptorSet descriptorWrite {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = dstSet;
	descriptorWrite.dstBinding = dstBinding;
	descriptorWrite.descriptorType = descriptorType;
	descriptorWrite.descriptorCount = descriptorCount;
	descriptorWrite.pBufferInfo = pBufferInfo;

	return descriptorWrite;
}

VkPipelineShaderStageCreateInfo load_shader(const std::string shader_name, VkShaderModule *shaderModule){
	const std::string shadersPath = "./";

	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

    shaderStage.module = tools::loadShader((shadersPath + shader_name).c_str(), singleton.device);
	shaderStage.pName = "main";
	//shaderStage.pSpecializationInfo = &specializationInfo;
	*shaderModule = shaderStage.module;

	return shaderStage;
}


void create_pipeline(VkPipelineShaderStageCreateInfo *shaderStage, VkPipelineLayout *pipelineLayout, VkPipeline *pipeline){
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(singleton.device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);
	// Create pipeline
	VkComputePipelineCreateInfo computePipelineCreateInfo {};
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.layout = *pipelineLayout;
	computePipelineCreateInfo.flags = 0;

	computePipelineCreateInfo.stage = *shaderStage;
	vkCreateComputePipelines(singleton.device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, pipeline);

}

void create_descriptor_pool(std::vector<VkDescriptorPoolSize> poolSizes, uint32_t maxSets){

			VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = maxSets;
			vkCreateDescriptorPool(singleton.device, &descriptorPoolInfo, nullptr, &descriptorPool);

}


void create_storage_buffer(const VkDeviceSize bufferSize, void* data, VkBuffer* device_buffer, VkDeviceMemory* device_memory, VkBuffer* host_buffer, VkDeviceMemory* host_memory){
		// Copy input data to VRAM using a staging buffer
	
			singleton.createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				host_buffer,
				host_memory,
				bufferSize,
				data);

			// Flush writes to host visible buffer
			void* mapped;
			vkMapMemory(singleton.device, *host_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = *host_memory;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkFlushMappedMemoryRanges(singleton.device, 1, &mappedRange);
			vkUnmapMemory(singleton.device, *host_memory);
			
			singleton.createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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
			vkAllocateCommandBuffers(singleton.device, &cmdBufAllocateInfo, &copyCmd);
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
			vkCreateFence(singleton.device, &fenceInfo, nullptr, &fence);

			// Submit to the queue
			vkQueueSubmit(singleton.queues[0], 1, &submitInfo, fence);
			vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);

			vkDestroyFence(singleton.device, fence, nullptr);
			vkFreeCommandBuffers(singleton.device, commandPool, 1, &copyCmd);
			
			vkMapMemory(singleton.device, *device_memory, 0, bufferSize, 0, &mapped);
			memcpy(mapped, data, bufferSize);
}




void create_shared_storage_buffer(const VkDeviceSize bufferSize, VkBuffer* device_buffer, VkDeviceMemory* device_memory, void* data, void** mapped){
		// Copy input data to VRAM using a staging buffer
			
			singleton.createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				device_buffer,
				device_memory,
				bufferSize,
				data);
			//void* mapped_temp;
			if (vkMapMemory(singleton.device, *device_memory, 0, bufferSize, 0, mapped) == VK_ERROR_MEMORY_MAP_FAILED ){
				std::cout<<"memory map failed"<<std::endl;
			}


			vkMapMemory(singleton.device, *device_memory, 0, bufferSize, 0, mapped);
			//memcpy(*mapped, data, bufferSize);
			
}


void create_shared_empty_storage_buffer(const VkDeviceSize bufferSize, VkBuffer* device_buffer, VkDeviceMemory* device_memory, void** mapped){
		// Copy input data to VRAM using a staging buffer
			
			singleton.createBuffer(
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				device_buffer,
				device_memory,
				bufferSize);
			//void* mapped_temp;
			if (vkMapMemory(singleton.device, *device_memory, 0, bufferSize, 0, mapped) == VK_ERROR_MEMORY_MAP_FAILED ){
				std::cout<<"memory map failed"<<std::endl;
			}


			//vkMapMemory(singleton.device, *device_memory, 0, bufferSize, 0, mapped);
			
}



VkBufferMemoryBarrier create_buffer_barrier(VkBuffer* buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask){
	VkBufferMemoryBarrier bufferBarrier {};
	bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferBarrier.buffer = *buffer;
	bufferBarrier.size = VK_WHOLE_SIZE;
	bufferBarrier.srcAccessMask = srcAccessMask;
	bufferBarrier.dstAccessMask = dstAccessMask;
	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	return bufferBarrier;
}

void create_pipeline_barrier(VkBufferMemoryBarrier* bufferBarrier, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask){
	vkCmdPipelineBarrier(
		commandBuffer,
		srcAccessMask,
		dstAccessMask,
		VK_FLAGS_NONE,
		0, nullptr,
		1, bufferBarrier,
		0, nullptr);
}

void create_fence(){
	VkFenceCreateInfo fenceCreateInfo {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	vkCreateFence(singleton.device, &fenceCreateInfo, nullptr, &fence);
}
};