#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define BUFFER_ELEMENTS 131072
#define PARTITION_SIZE 7680
#define BINNING_THREAD_BLOCKS  (BUFFER_ELEMENTS + PARTITION_SIZE - 1) / PARTITION_SIZE

class Init : public ApplicationBase{
    public:
    Init() : ApplicationBase() {};
    ~Init() {};
	void 		submit();
	void 		cleanup(VkPipeline *pipeline);
	void 		run(const int blocks, glm::vec4* data, const int n, const int min_val, const float range, const float seed);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
        int size;
        int min_val;
        int range;
        int seed;
	} init_val_push_constant;
    
	struct{
		VkBuffer data_buffer;
	} buffer;

	struct{
		VkBuffer data_buffer;
	} temp_buffer;

	struct{
		VkDeviceMemory data_memory;
	} memory;

	struct{
		VkDeviceMemory data_memory;
	} temp_memory;

};


void Init::submit(){
			// todo: change the harded coded for map
			printf("execute\n");
			vkResetFences(singleton.device, 1, &fence);
			const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &computeSubmitInfo, fence);
			vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);

}

void Init::cleanup(VkPipeline *pipeline){
		
		vkDestroyBuffer(singleton.device, buffer.data_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.data_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.data_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.data_memory, nullptr);

		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void Init::run(const int blocks, glm::vec4* data, const int n, const int min_val, const float range, const float seed){
	 
	
	VkPipeline pipeline;


	create_storage_buffer(n*sizeof(glm::vec4), data, &buffer.data_buffer, &memory.data_memory, &temp_buffer.data_buffer, &temp_memory.data_memory);
	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
	VkDescriptorSetLayoutBinding b_data_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
		b_data_layoutBinding
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
	
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		data_descriptor_write
	};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("init.spv", &shaderModule);
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

	create_pipeline_barrier(&data_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	init_val_push_constant.min_val = min_val;
	init_val_push_constant.range = range;
	init_val_push_constant.seed = seed;
    init_val_push_constant.size = n;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &init_val_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, blocks, 1, 1);
	
	data_barrier = create_buffer_barrier(&buffer.data_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&data_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferCopy copyRegion = {};
	copyRegion.size = n* sizeof(glm::vec4);
	vkCmdCopyBuffer(commandBuffer, buffer.data_buffer, temp_buffer.data_buffer, 1, &copyRegion);
	data_barrier = create_buffer_barrier(&buffer.data_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&data_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);


	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit();

	// Make device writes visible to the host
	void *mapped;
	vkMapMemory(singleton.device,temp_memory.data_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = temp_memory.data_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
			
	// Copy to output
	const VkDeviceSize bufferSize = n * sizeof(glm::vec4);
	memcpy(data, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.data_memory);

	vkQueueWaitIdle(queue);

	cleanup(&pipeline);
	
	    for (int i = 0; i < 1024; ++i){
        std::cout << data[i].x << " " << data[i].y << " " << data[i].z << " " << data[i].w << std::endl;
    }
}

