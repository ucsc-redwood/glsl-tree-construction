#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define PARTITION_SIZE 3072

class PrefixSum : public ApplicationBase{
    public:
    PrefixSum() : ApplicationBase() {};
    ~PrefixSum() {};
    void        submit(const int queue_idx);
	void 		 cleanup(VkPipeline *prefix_sum_pipeline);
	void run(const int logical_block,
	const int queue_idx,
	uint32_t *u_keys,
	volatile uint32_t *reduction,
	volatile uint32_t *index,
	VkBuffer u_keys_buffer,
	VkBuffer reduction_buffer,
	VkBuffer index_buffer,
	const int n);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};
	struct PushConstant {
		uint32_t aligned_size;
	} prefix_sum_push_constant;

};


void PrefixSum::submit(const int queue_idx){
			// todo: change the harded coded for map
			vkResetFences(singleton.device, 1, &fence);
			const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(singleton.queues[queue_idx], 1, &computeSubmitInfo, fence);
			vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);

}

void PrefixSum::cleanup(VkPipeline *pipeline){

		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void PrefixSum::run(const int logical_block,
const int queue_idx,
uint32_t *u_keys,
volatile uint32_t *reduction,
volatile uint32_t *index,
VkBuffer u_keys_buffer,
VkBuffer reduction_buffer,
VkBuffer index_buffer,
const int n){

	const uint32_t aligned_size = ((n + 4 - 1)/ 4) * 4;
	const uint32_t vectorized_size = aligned_size / 4;
	const uint32_t num_blocks = (aligned_size + PARTITION_SIZE - 1) / PARTITION_SIZE;

    VkPipeline pipeline;

	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
    VkDescriptorSetLayoutBinding b_u_keys_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
    VkDescriptorSetLayoutBinding b_reduction_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
    VkDescriptorSetLayoutBinding b_index_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
        b_u_keys_layoutBinding, b_reduction_layoutBinding, b_index_layoutBinding
	};
	// create descriptor 
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
    VkDescriptorBufferInfo u_keys_bufferDescriptor = { u_keys_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet u_keys_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &u_keys_bufferDescriptor);
    VkDescriptorBufferInfo reduction_bufferDescriptor = { reduction_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet reduction_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &reduction_bufferDescriptor);
    VkDescriptorBufferInfo index_bufferDescriptor = { index_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet index_descriptor_write = create_descriptor_write(descriptorSets[0],2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &index_bufferDescriptor);


	

	std::vector<VkWriteDescriptorSet> descriptor_writes = {
        u_keys_descriptor_write, reduction_descriptor_write, index_descriptor_write
    };
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);

	//create pipeline 
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("prefix_sum.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
    VkBufferMemoryBarrier u_keys_barrier = create_buffer_barrier(&u_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier reduction_barrier = create_buffer_barrier(&reduction_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier index_barrier = create_buffer_barrier(&index_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    create_pipeline_barrier(&u_keys_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&reduction_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&index_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	prefix_sum_push_constant.aligned_size = vectorized_size;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &prefix_sum_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, num_blocks, 1, 1);

    u_keys_barrier = create_buffer_barrier(&u_keys_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    create_pipeline_barrier(&u_keys_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit(queue_idx);



	vkQueueWaitIdle(singleton.queues[queue_idx]);


	cleanup(&pipeline);
}

