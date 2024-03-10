#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define BUFFER_ELEMENTS 131072

class EdgeCount : public ApplicationBase{
    public:
    EdgeCount() : ApplicationBase() {};
    ~EdgeCount() {};
	void 		submit();
	void 		cleanup(VkPipeline *pipeline);
	void 		run(const int logical_blocks, uint8_t* prefix_n, int* parent, uint32_t* edge_count, int n_brt_nodes);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
		int n_brt_nodes;
	} edge_count_push_constant;
    
	struct{
		VkBuffer prefix_n_buffer;
		VkBuffer parent_buffer;
		VkBuffer edge_count_buffer;

	} buffer;

	struct{
		VkBuffer prefix_n_buffer;
		VkBuffer parent_buffer;
		VkBuffer edge_count_buffer;
	} temp_buffer;

	struct{
		VkDeviceMemory prefix_n_memory;
		VkDeviceMemory parent_memory;
		VkDeviceMemory edge_count_memory;
	} memory;

	struct{
		VkDeviceMemory prefix_n_memory;
		VkDeviceMemory parent_memory;
		VkDeviceMemory edge_count_memory;

	} temp_memory;

};


void EdgeCount::submit(){
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

void EdgeCount::cleanup(VkPipeline *pipeline){
		

		vkDestroyBuffer(singleton.device, buffer.prefix_n_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.prefix_n_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.prefix_n_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.prefix_n_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.parent_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.parent_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.parent_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.parent_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.edge_count_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.edge_count_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.edge_count_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.edge_count_memory, nullptr);

		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void EdgeCount::run(const int logical_blocks, uint8_t* prefix_n, int* parent, uint32_t* edge_count, int n_brt_nodes){
	 
	
	VkPipeline pipeline;

	create_storage_buffer(n_brt_nodes*sizeof(uint8_t), prefix_n, &buffer.prefix_n_buffer, &memory.prefix_n_memory, &temp_buffer.prefix_n_buffer, &temp_memory.prefix_n_memory);
	create_storage_buffer(n_brt_nodes*sizeof(int), parent, &buffer.parent_buffer, &memory.parent_memory, &temp_buffer.parent_buffer, &temp_memory.parent_memory);
	create_storage_buffer(n_brt_nodes*sizeof(uint32_t), edge_count, &buffer.edge_count_buffer, &memory.edge_count_memory, &temp_buffer.edge_count_buffer, &temp_memory.edge_count_memory);

	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
	VkDescriptorSetLayoutBinding b_prefix_n_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_parent_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding b_edge_count_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
		b_prefix_n_layoutBinding, b_parent_layoutBinding, b_edge_count_layoutBinding
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

	VkDescriptorBufferInfo prefix_n_bufferDescriptor = { buffer.prefix_n_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet prefix_n_descriptor_write = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &prefix_n_bufferDescriptor);
	VkDescriptorBufferInfo parent_bufferDescriptor = { buffer.parent_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet parent_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &parent_bufferDescriptor);
	VkDescriptorBufferInfo edge_count_bufferDescriptor = { buffer.edge_count_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet edge_count_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &edge_count_bufferDescriptor);
	
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		prefix_n_descriptor_write, parent_descriptor_write, edge_count_descriptor_write
	};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("edge_count.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);



	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier prefix_n_barrier = create_buffer_barrier(&buffer.prefix_n_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier parent_barrier = create_buffer_barrier(&buffer.parent_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier edge_count_barrier = create_buffer_barrier(&buffer.edge_count_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);


	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&edge_count_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	edge_count_push_constant.n_brt_nodes = n_brt_nodes;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &edge_count_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	
	prefix_n_barrier = create_buffer_barrier(&buffer.prefix_n_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	parent_barrier = create_buffer_barrier(&buffer.parent_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	edge_count_barrier = create_buffer_barrier(&buffer.edge_count_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&edge_count_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


	VkBufferCopy edge_count_copyRegion = {};
	edge_count_copyRegion.size = n_brt_nodes* sizeof(uint32_t);
	vkCmdCopyBuffer(commandBuffer, buffer.edge_count_buffer, temp_buffer.edge_count_buffer, 1, &edge_count_copyRegion);
	edge_count_barrier = create_buffer_barrier(&buffer.edge_count_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&edge_count_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

	vkEndCommandBuffer(commandBuffer);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit();

	// Make device writes visible to the host
	void *mapped;
	vkMapMemory(singleton.device,temp_memory.edge_count_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = temp_memory.edge_count_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	VkDeviceSize bufferSize = n_brt_nodes * sizeof(uint32_t);
	memcpy(edge_count, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.edge_count_memory);


	vkQueueWaitIdle(singleton.queue);

	cleanup(&pipeline);
}

