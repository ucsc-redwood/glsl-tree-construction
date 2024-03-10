#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define BUFFER_ELEMENTS 131072

class RadixTree : public ApplicationBase{
    public:
    RadixTree() : ApplicationBase() {};
    ~RadixTree() {};
	void 		submit();
	void 		cleanup(VkPipeline *pipeline);
	void 		run(const int logical_blocks, uint32_t* morton_keys, uint8_t* prefix_n, bool* has_leaf_left, bool* has_leaf_right, int* left_child, int* parent, const int n_unique);

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
		int n;
	} radix_tree_push_constant;
    
	struct{
		VkBuffer morton_keys_buffer;
		VkBuffer prefix_n_buffer;
		VkBuffer has_leaf_left_buffer;
		VkBuffer has_leaf_right_buffer;
		VkBuffer left_child_buffer;
		VkBuffer parent_buffer;
	} buffer;

	struct{
		VkBuffer morton_keys_buffer;
		VkBuffer prefix_n_buffer;
		VkBuffer has_leaf_left_buffer;
		VkBuffer has_leaf_right_buffer;
		VkBuffer left_child_buffer;
		VkBuffer parent_buffer;
	} temp_buffer;

	struct{
		VkDeviceMemory morton_keys_memory;
		VkDeviceMemory prefix_n_memory;
		VkDeviceMemory has_leaf_left_memory;
		VkDeviceMemory has_leaf_right_memory;
		VkDeviceMemory left_child_memory;
		VkDeviceMemory parent_memory;
	} memory;

	struct{
		VkDeviceMemory morton_keys_memory;
		VkDeviceMemory prefix_n_memory;
		VkDeviceMemory has_leaf_left_memory;
		VkDeviceMemory has_leaf_right_memory;
		VkDeviceMemory left_child_memory;
		VkDeviceMemory parent_memory;
	} temp_memory;

};


void RadixTree::submit(){
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

void RadixTree::cleanup(VkPipeline *pipeline){
		
		vkDestroyBuffer(singleton.device, buffer.morton_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.morton_keys_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.morton_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.morton_keys_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.prefix_n_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.prefix_n_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.prefix_n_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.prefix_n_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.has_leaf_left_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.has_leaf_left_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.has_leaf_left_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.has_leaf_left_memory, nullptr);
		
		vkDestroyBuffer(singleton.device, buffer.has_leaf_right_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.has_leaf_right_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.has_leaf_right_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.has_leaf_right_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.left_child_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.left_child_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.left_child_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.left_child_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.parent_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.parent_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.parent_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.parent_memory, nullptr);




		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void RadixTree::run(const int logical_blocks, uint32_t* morton_keys, uint8_t* prefix_n, bool* has_leaf_left, bool* has_leaf_right, int* left_child, int* parent, const int n_unique){
	 
	
	VkPipeline pipeline;

	int n = n_unique - 1;
	create_storage_buffer(n_unique*sizeof(uint32_t), morton_keys, &buffer.morton_keys_buffer, &memory.morton_keys_memory, &temp_buffer.morton_keys_buffer, &temp_memory.morton_keys_memory);
	create_storage_buffer(n*sizeof(uint8_t), prefix_n, &buffer.prefix_n_buffer, &memory.prefix_n_memory, &temp_buffer.prefix_n_buffer, &temp_memory.prefix_n_memory);
	create_storage_buffer(n*sizeof(bool), has_leaf_left, &buffer.has_leaf_left_buffer, &memory.has_leaf_left_memory, &temp_buffer.has_leaf_left_buffer, &temp_memory.has_leaf_left_memory);
	create_storage_buffer(n*sizeof(bool), has_leaf_right, &buffer.has_leaf_right_buffer, &memory.has_leaf_right_memory, &temp_buffer.has_leaf_right_buffer, &temp_memory.has_leaf_right_memory);
	create_storage_buffer(n*sizeof(int), left_child, &buffer.left_child_buffer, &memory.left_child_memory, &temp_buffer.left_child_buffer, &temp_memory.left_child_memory);
	create_storage_buffer(n*sizeof(int), parent, &buffer.parent_buffer, &memory.parent_memory, &temp_buffer.parent_buffer, &temp_memory.parent_memory);
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
	VkDescriptorBufferInfo morton_keys_bufferDescriptor = { buffer.morton_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet morton_keys_descriptor_write = create_descriptor_write(descriptorSets[0],0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &morton_keys_bufferDescriptor);
	VkDescriptorBufferInfo prefix_n_bufferDescriptor = { buffer.prefix_n_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet prefix_n_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &prefix_n_bufferDescriptor);
	VkDescriptorBufferInfo has_leaf_left_bufferDescriptor = { buffer.has_leaf_left_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet has_leaf_left_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &has_leaf_left_bufferDescriptor);
	VkDescriptorBufferInfo has_leaf_right_bufferDescriptor = { buffer.has_leaf_right_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet has_leaf_right_descriptor_write = create_descriptor_write(descriptorSets[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &has_leaf_right_bufferDescriptor);
	VkDescriptorBufferInfo left_child_bufferDescriptor = { buffer.left_child_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet left_child_descriptor_write = create_descriptor_write(descriptorSets[0], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &left_child_bufferDescriptor);
	VkDescriptorBufferInfo parent_bufferDescriptor = { buffer.parent_buffer, 0, VK_WHOLE_SIZE };
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
	VkBufferMemoryBarrier morton_keys_barrier = create_buffer_barrier(&buffer.morton_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier prefix_n_barrier = create_buffer_barrier(&buffer.prefix_n_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier has_leaf_left_barrier = create_buffer_barrier(&buffer.has_leaf_left_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier has_leaf_right_barrier = create_buffer_barrier(&buffer.has_leaf_right_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier left_child_barrier = create_buffer_barrier(&buffer.left_child_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier parent_barrier = create_buffer_barrier(&buffer.parent_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

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
	
	prefix_n_barrier = create_buffer_barrier(&buffer.prefix_n_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	has_leaf_left_barrier = create_buffer_barrier(&buffer.has_leaf_left_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	has_leaf_right_barrier = create_buffer_barrier(&buffer.has_leaf_right_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	left_child_barrier = create_buffer_barrier(&buffer.left_child_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	parent_barrier = create_buffer_barrier(&buffer.parent_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&has_leaf_left_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&has_leaf_right_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&left_child_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferCopy prefix_n_copyRegion = {};
	prefix_n_copyRegion.size = n* sizeof(uint8_t);
	VkBufferCopy has_leaf_left_copyRegion = {};
	has_leaf_left_copyRegion.size = n* sizeof(bool);
	VkBufferCopy has_leaf_right_copyRegion = {};
	has_leaf_right_copyRegion.size = n* sizeof(bool);
	VkBufferCopy left_child_copyRegion = {};
	left_child_copyRegion.size = n* sizeof(int);
	VkBufferCopy parent_copyRegion = {};
	parent_copyRegion.size = n* sizeof(int);
	vkCmdCopyBuffer(commandBuffer, buffer.prefix_n_buffer, temp_buffer.prefix_n_buffer, 1, &prefix_n_copyRegion);
	vkCmdCopyBuffer(commandBuffer, buffer.has_leaf_left_buffer, temp_buffer.has_leaf_left_buffer, 1, &has_leaf_left_copyRegion);
	vkCmdCopyBuffer(commandBuffer, buffer.has_leaf_right_buffer, temp_buffer.has_leaf_right_buffer, 1, &has_leaf_right_copyRegion);
	vkCmdCopyBuffer(commandBuffer, buffer.left_child_buffer, temp_buffer.left_child_buffer, 1, &left_child_copyRegion);
	vkCmdCopyBuffer(commandBuffer, buffer.parent_buffer, temp_buffer.parent_buffer, 1, &parent_copyRegion);


	prefix_n_barrier = create_buffer_barrier(&buffer.prefix_n_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	has_leaf_left_barrier = create_buffer_barrier(&buffer.has_leaf_left_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	has_leaf_right_barrier = create_buffer_barrier(&buffer.has_leaf_right_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	left_child_barrier = create_buffer_barrier(&buffer.left_child_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	parent_barrier = create_buffer_barrier(&buffer.parent_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&prefix_n_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
	create_pipeline_barrier(&has_leaf_left_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
	create_pipeline_barrier(&has_leaf_right_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
	create_pipeline_barrier(&left_child_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
	create_pipeline_barrier(&parent_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

	vkEndCommandBuffer(commandBuffer);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit();

	// Make device writes visible to the host
	void *mapped;
	vkMapMemory(singleton.device,temp_memory.prefix_n_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = temp_memory.prefix_n_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	VkDeviceSize bufferSize = n * sizeof(uint8_t);
	memcpy(prefix_n, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.prefix_n_memory);

	vkMapMemory(singleton.device,temp_memory.has_leaf_left_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	mappedRange.memory = temp_memory.has_leaf_left_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	bufferSize = n * sizeof(bool);
	memcpy(has_leaf_left, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.has_leaf_left_memory);

	vkMapMemory(singleton.device,temp_memory.has_leaf_right_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	mappedRange.memory = temp_memory.has_leaf_right_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	bufferSize = n * sizeof(bool);
	memcpy(has_leaf_right, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.has_leaf_right_memory);

	vkMapMemory(singleton.device,temp_memory.left_child_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	mappedRange.memory = temp_memory.left_child_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	bufferSize = n * sizeof(int);
	memcpy(left_child, mapped, bufferSize);
	vkUnmapMemory(singleton.device,temp_memory.left_child_memory);

	vkMapMemory(singleton.device,temp_memory.parent_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	mappedRange.memory = temp_memory.parent_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	bufferSize = n * sizeof(int);
	memcpy(parent, mapped, bufferSize);

	vkQueueWaitIdle(singleton.queue);

	cleanup(&pipeline);
}

