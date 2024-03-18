#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>


class RadixTree : public ApplicationBase{
    public:
    RadixTree() : ApplicationBase() {};
    ~RadixTree() {};
	void 		submit(const int queue_idx);
	void 		cleanup(VkPipeline *pipeline);
	void 		run(const int logical_blocks,
	const int queue_idx,
	uint32_t* morton_keys,
	uint8_t* prefix_n,
	bool* has_leaf_left,
	bool* has_leaf_right,
	int* left_child,
	int* parent,
	VkBuffer morton_keys_buffer,
	VkBuffer prefix_n_buffer,
	VkBuffer has_leaf_left_buffer,
	VkBuffer has_leaf_right_buffer,
	VkBuffer left_child_buffer,
	VkBuffer parent_buffer,
	const int n_unique);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
		int n;
	} radix_tree_push_constant;
   

};


void RadixTree::submit(const int queue_idx){
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

void RadixTree::cleanup(VkPipeline *pipeline){


		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void RadixTree::run(const int logical_blocks,
const int queue_idx,
uint32_t* morton_keys,
uint8_t* prefix_n,
bool* has_leaf_left,
bool* has_leaf_right,
int* left_child,
int* parent,
VkBuffer morton_keys_buffer,
VkBuffer prefix_n_buffer,
VkBuffer has_leaf_left_buffer,
VkBuffer has_leaf_right_buffer,
VkBuffer left_child_buffer,
VkBuffer parent_buffer,
 const int n_unique){
	 
	
	VkPipeline pipeline;

	int n = n_unique - 1;
	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
	VkDescriptorSetLayoutBinding b_morton_keys_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_prefix_n_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding b_has_leaf_left_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
	VkDescriptorSetLayoutBinding b_has_leaf_right_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);
	VkDescriptorSetLayoutBinding b_left_child_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1);
	VkDescriptorSetLayoutBinding b_parent_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
		b_morton_keys_layoutBinding, b_prefix_n_layoutBinding, b_has_leaf_left_layoutBinding, b_has_leaf_right_layoutBinding, b_left_child_layoutBinding, b_parent_layoutBinding
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
	VkDescriptorBufferInfo morton_keys_bufferDescriptor = { morton_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet morton_keys_descriptor_write = create_descriptor_write(descriptorSets[0],0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &morton_keys_bufferDescriptor);
	VkDescriptorBufferInfo prefix_n_bufferDescriptor = { prefix_n_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet prefix_n_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &prefix_n_bufferDescriptor);
	VkDescriptorBufferInfo has_leaf_left_bufferDescriptor = { has_leaf_left_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet has_leaf_left_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &has_leaf_left_bufferDescriptor);
	VkDescriptorBufferInfo has_leaf_right_bufferDescriptor = { has_leaf_right_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet has_leaf_right_descriptor_write = create_descriptor_write(descriptorSets[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &has_leaf_right_bufferDescriptor);
	VkDescriptorBufferInfo left_child_bufferDescriptor = { left_child_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet left_child_descriptor_write = create_descriptor_write(descriptorSets[0], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &left_child_bufferDescriptor);
	VkDescriptorBufferInfo parent_bufferDescriptor = { parent_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet parent_descriptor_write = create_descriptor_write(descriptorSets[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &parent_bufferDescriptor);

	
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		morton_keys_descriptor_write, prefix_n_descriptor_write, has_leaf_left_descriptor_write, has_leaf_right_descriptor_write, left_child_descriptor_write, parent_descriptor_write
	};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("build_radix_tree.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);



	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier morton_keys_barrier = create_buffer_barrier(&morton_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier prefix_n_barrier = create_buffer_barrier(&prefix_n_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier has_leaf_left_barrier = create_buffer_barrier(&has_leaf_left_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier has_leaf_right_barrier = create_buffer_barrier(&has_leaf_right_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier left_child_barrier = create_buffer_barrier(&left_child_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier parent_barrier = create_buffer_barrier(&parent_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	create_pipeline_barrier(&morton_keys_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&has_leaf_left_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&has_leaf_right_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&left_child_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	radix_tree_push_constant.n = n;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &radix_tree_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	
	prefix_n_barrier = create_buffer_barrier(&prefix_n_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	has_leaf_left_barrier = create_buffer_barrier(&has_leaf_left_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	has_leaf_right_barrier = create_buffer_barrier(&has_leaf_right_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	left_child_barrier = create_buffer_barrier(&left_child_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	parent_barrier = create_buffer_barrier(&parent_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&has_leaf_left_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&has_leaf_right_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&left_child_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	vkEndCommandBuffer(commandBuffer);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit(queue_idx);


	vkQueueWaitIdle(singleton.queues[queue_idx]);

	cleanup(&pipeline);
}

