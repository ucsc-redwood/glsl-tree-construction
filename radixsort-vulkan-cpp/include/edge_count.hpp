#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

class EdgeCount : public ApplicationBase
{
public:
	EdgeCount() : ApplicationBase(){};
	~EdgeCount(){};
	void submit(const int queue_idx);
	void cleanup(VkPipeline *pipeline);
	void run(const int logical_blocks,
			 const int queue_idx,
			 uint8_t *prefix_n,
			 int *parent,
			 int *edge_count,
			 VkBuffer prefix_n_buffer,
			 VkBuffer parent_buffer,
			 VkBuffer edge_count_buffer,
			 int n_brt_nodes);

	double time();

private:
	VkShaderModule shaderModule;

	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};
	double run_time;
	struct PushConstant
	{
		int n_brt_nodes;
	} edge_count_push_constant;
};

void EdgeCount::submit(const int queue_idx)
{
	vkResetFences(singleton.device, 1, &fence);
	const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
	VkSubmitInfo computeSubmitInfo{};
	computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(singleton.queues[queue_idx], 1, &computeSubmitInfo, fence);
	vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);
}

void EdgeCount::cleanup(VkPipeline *pipeline)
{

	vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
	vkDestroyPipeline(singleton.device, *pipeline, nullptr);
	vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
}

void EdgeCount::run(const int logical_blocks,
					const int queue_idx,
					uint8_t *prefix_n,
					int *parent,
					int *edge_count,
					VkBuffer prefix_n_buffer,
					VkBuffer parent_buffer,
					VkBuffer edge_count_buffer,
					int n_brt_nodes)
{

	VkPipeline pipeline;

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
		b_prefix_n_layoutBinding, b_parent_layoutBinding, b_edge_count_layoutBinding};

	// create descriptor set layout for both histogram and binning
	create_descriptor_set_layout(set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(1, descriptorSetLayouts);
	// add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(1, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind

	VkDescriptorBufferInfo prefix_n_bufferDescriptor = {prefix_n_buffer, 0, VK_WHOLE_SIZE};
	VkWriteDescriptorSet prefix_n_descriptor_write = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &prefix_n_bufferDescriptor);
	VkDescriptorBufferInfo parent_bufferDescriptor = {parent_buffer, 0, VK_WHOLE_SIZE};
	VkWriteDescriptorSet parent_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &parent_bufferDescriptor);
	VkDescriptorBufferInfo edge_count_bufferDescriptor = {edge_count_buffer, 0, VK_WHOLE_SIZE};
	VkWriteDescriptorSet edge_count_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &edge_count_bufferDescriptor);

	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		prefix_n_descriptor_write, parent_descriptor_write, edge_count_descriptor_write};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);

	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("edge_count.spv", &shaderModule);
	create_pipeline(&shader_stage, &pipelineLayout, &pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);

	// record command buffer, which involves binding the pipeline and descriptor sets,
	// specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo{};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	vkCmdResetQueryPool(commandBuffer, singleton.query_pool_timestamps, 0, 2);
	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, singleton.query_pool_timestamps, 0);
	VkBufferMemoryBarrier prefix_n_barrier = create_buffer_barrier(&prefix_n_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier parent_barrier = create_buffer_barrier(&parent_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier edge_count_barrier = create_buffer_barrier(&edge_count_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&edge_count_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	edge_count_push_constant.n_brt_nodes = n_brt_nodes;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &edge_count_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);

	prefix_n_barrier = create_buffer_barrier(&prefix_n_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	parent_barrier = create_buffer_barrier(&parent_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	edge_count_barrier = create_buffer_barrier(&edge_count_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&edge_count_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, singleton.query_pool_timestamps, 1);
	vkEndCommandBuffer(commandBuffer);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit(queue_idx);

	vkQueueWaitIdle(singleton.queues[queue_idx]);

	uint64_t timestamps[2];
	vkGetQueryPoolResults(singleton.device, singleton.query_pool_timestamps, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

	uint64_t elapsedTimeNs = timestamps[1] - timestamps[0];
	double elapsedTimeMs = elapsedTimeNs / 1000000.0;
	double elapsedTimeSec = elapsedTimeNs / 100000000.0;
	// std::cout << "Elapsed time morton: " << elapsedTimeMs << "ms" << std::endl;
	run_time = elapsedTimeSec;

	cleanup(&pipeline);
}

double EdgeCount::time()
{
	return run_time;
}
