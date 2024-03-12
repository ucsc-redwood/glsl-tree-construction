#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

class Morton : public ApplicationBase{
    public:
    Morton() : ApplicationBase() {};
    ~Morton() {};
	void 		submit();
	void 		cleanup(VkPipeline *pipeline);
	void 		run( const int logical_blocks, glm::vec4* data, uint32_t* morton_keys, const uint32_t n, const float min_coord, const float range);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
        uint32_t n;
		float min_coord;
		float range;
	} morton_push_constant;
    
	struct{
		VkBuffer data_buffer;
		VkBuffer morton_keys_buffer;
	} buffer;

	struct{
		VkBuffer data_buffer;
		VkBuffer morton_keys_buffer;
	} temp_buffer;

	struct{
		VkDeviceMemory data_memory;
		VkDeviceMemory morton_keys_memory;
	} memory;

	struct{
		VkDeviceMemory data_memory;
		VkDeviceMemory morton_keys_memory;
	} temp_memory;

};


void Morton::submit(){
			// todo: change the harded coded for map
			printf("execute\n");
			vkResetFences(singleton.device, 1, &fence);
			const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(singleton.queue, 1, &computeSubmitInfo, fence);
			vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);

}

void Morton::cleanup(VkPipeline *pipeline){
		
		vkDestroyBuffer(singleton.device, buffer.data_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.data_memory, nullptr);
		vkDestroyBuffer(singleton.device, buffer.morton_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.morton_keys_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.data_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.data_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.morton_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.morton_keys_memory, nullptr);



		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void Morton::run(const int logical_blocks, glm::vec4* data, uint32_t* morton_keys, const uint32_t n, const float min_coord, const float range){
	 
	
	VkPipeline pipeline;


	create_storage_buffer(n*sizeof(uint32_t), morton_keys, &buffer.morton_keys_buffer, &memory.morton_keys_memory, &temp_buffer.morton_keys_buffer, &temp_memory.morton_keys_memory);
	create_storage_buffer(n*sizeof(glm::vec4), data, &buffer.data_buffer, &memory.data_memory, &temp_buffer.data_buffer, &temp_memory.data_memory);
	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
	VkDescriptorSetLayoutBinding b_data_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_morton_keys_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
		b_morton_keys_layoutBinding, b_data_layoutBinding
	};

	// create descriptor set layout for both histogram and binning
	create_descriptor_set_layout(set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(1, descriptorSetLayouts);
	//add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(1, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
	VkDescriptorBufferInfo data_bufferDescriptor = { buffer.data_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet data_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &data_bufferDescriptor);
	VkDescriptorBufferInfo morton_keys_bufferDescriptor = { buffer.morton_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet morton_keys_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &morton_keys_bufferDescriptor);
	
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		data_descriptor_write, morton_keys_descriptor_write
	};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("morton.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);



	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier data_barrier = create_buffer_barrier(&buffer.data_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier morton_keys_barrier = create_buffer_barrier(&buffer.morton_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	create_pipeline_barrier(&data_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&morton_keys_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	morton_push_constant.min_coord = min_coord;
	morton_push_constant.range = range;
	morton_push_constant.n = n;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &morton_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	
	morton_keys_barrier = create_buffer_barrier(&buffer.morton_keys_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&morton_keys_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferCopy copyRegion = {};
	copyRegion.size = n* sizeof(uint32_t);
	vkCmdCopyBuffer(commandBuffer, buffer.morton_keys_buffer, temp_buffer.morton_keys_buffer, 1, &copyRegion);
	morton_keys_barrier = create_buffer_barrier(&buffer.morton_keys_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&morton_keys_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);


	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit();

	// Make device writes visible to the host
	void *mapped;
	vkMapMemory(singleton.device,temp_memory.morton_keys_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = temp_memory.morton_keys_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
			
	const VkDeviceSize bufferSize = n * sizeof(uint32_t);
	memcpy(morton_keys, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.morton_keys_memory);

	vkQueueWaitIdle(singleton.queue);

	cleanup(&pipeline);
}

