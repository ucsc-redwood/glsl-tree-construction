#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define PARTITION_SIZE 3072

class Unique : public ApplicationBase{
    public:
    Unique() : ApplicationBase() {};
    ~Unique() {};
    void        execute();
	void 		 cleanup(VkPipeline *find_dup_pipeline, VkPipeline *prefix_sum_pipeline, VkPipeline *move_dup_pipeline);
	void run(const int logical_block, uint32_t* sorted_keys, uint32_t* u_keys, uint32_t* contributions, const int n);

    private:
	VkShaderModule find_dups_shaderModule;
	VkShaderModule prefix_sum_shaderModule;
	VkShaderModule move_dups_shaderModule;

	VkDescriptorSetLayout descriptorSetLayouts[3] = {VkDescriptorSetLayout{}, VkDescriptorSetLayout{}, VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[3] = {VkDescriptorSet{}, VkDescriptorSet{}, VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[3] = {VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}};
	struct PushConstant {
		int n;
	} unique_push_constant;
	struct{
		VkBuffer sorted_keys_buffer;
		VkBuffer contributions_buffer;
		VkBuffer u_keys_buffer;
		VkBuffer index_buffer;
		VkBuffer reduction_buffer;
	} buffer;

	struct{
		VkBuffer sorted_keys_buffer;
		VkBuffer contributions_buffer;
		VkBuffer u_keys_buffer;
		VkBuffer index_buffer;
		VkBuffer reduction_buffer;
	} temp_buffer;

	struct{
		VkDeviceMemory sorted_keys_memory;
		VkDeviceMemory contributions_memory;
		VkDeviceMemory u_keys_memory;
		VkDeviceMemory index_memory;
		VkDeviceMemory reduction_memory;
	} memory;

	struct{
		VkDeviceMemory sorted_keys_memory;
		VkDeviceMemory contributions_memory;
		VkDeviceMemory u_keys_memory;
		VkDeviceMemory index_memory;
		VkDeviceMemory reduction_memory;
	} temp_memory;

};


void Unique::execute(){
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

void Unique::cleanup(VkPipeline *find_dup_pipeline, VkPipeline *prefix_sum_pipeline, VkPipeline *move_dup_pipeline){
        vkDestroyBuffer(singleton.device, buffer.u_keys_buffer, nullptr);
        vkFreeMemory(singleton.device, memory.u_keys_memory, nullptr);
        vkDestroyBuffer(singleton.device, temp_buffer.u_keys_buffer, nullptr);
        vkFreeMemory(singleton.device, temp_memory.u_keys_memory, nullptr);

        vkDestroyBuffer(singleton.device, buffer.reduction_buffer, nullptr);
        vkFreeMemory(singleton.device, memory.reduction_memory, nullptr);
        vkDestroyBuffer(singleton.device, temp_buffer.reduction_buffer, nullptr);
        vkFreeMemory(singleton.device, temp_memory.reduction_memory, nullptr);

        vkDestroyBuffer(singleton.device, buffer.index_buffer, nullptr);
        vkFreeMemory(singleton.device, memory.index_memory, nullptr);
        vkDestroyBuffer(singleton.device, temp_buffer.index_buffer, nullptr);
        vkFreeMemory(singleton.device, temp_memory.index_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.sorted_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.sorted_keys_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.sorted_keys_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.sorted_keys_memory, nullptr);

		vkDestroyBuffer(singleton.device, buffer.contributions_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.contributions_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.contributions_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.contributions_memory, nullptr);




		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[1], nullptr);
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[2], nullptr);
		vkDestroyPipeline(singleton.device, *find_dup_pipeline, nullptr);
		vkDestroyPipeline(singleton.device, *prefix_sum_pipeline, nullptr);
		vkDestroyPipeline(singleton.device, *move_dup_pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, find_dups_shaderModule, nullptr);
		vkDestroyShaderModule(singleton.device, prefix_sum_shaderModule, nullptr);
		vkDestroyShaderModule(singleton.device, move_dups_shaderModule, nullptr);
		
}

void Unique::run(const int logical_block, uint32_t* sorted_keys, uint32_t* u_keys, uint32_t* contributions, const int n){

    VkPipeline find_dup_pipeline;
	VkPipeline prefix_sum_pipeline;
	VkPipeline move_dup_pipeline;

	uint32_t aligned_size = ((n + 4 - 1)/ 4) * 4;
    uint32_t index[1] = {0};
    const uint32_t num_blocks = (aligned_size + PARTITION_SIZE - 1) / PARTITION_SIZE;
	std::vector<uint32_t> reduction(num_blocks, 0);

    create_storage_buffer(n*sizeof(uint32_t), sorted_keys, &buffer.sorted_keys_buffer, &memory.sorted_keys_memory, &temp_buffer.sorted_keys_buffer, &temp_memory.sorted_keys_memory);
    create_storage_buffer(n*sizeof(uint32_t), contributions, &buffer.contributions_buffer, &memory.contributions_memory, &temp_buffer.contributions_buffer, &temp_memory.contributions_memory);
	create_storage_buffer(n*sizeof(uint32_t), u_keys, &buffer.u_keys_buffer, &memory.u_keys_memory, &temp_buffer.u_keys_buffer, &temp_memory.u_keys_memory);
    create_storage_buffer(sizeof(uint32_t), index, &buffer.index_buffer, &memory.index_memory, &temp_buffer.index_buffer, &temp_memory.index_memory);
    create_storage_buffer(num_blocks*sizeof(uint32_t), reduction.data(), &buffer.reduction_buffer, &memory.reduction_memory, &temp_buffer.reduction_buffer, &temp_memory.reduction_memory);
    
	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
	};

	create_descriptor_pool(poolSizes, 3);

	// create layout binding
    VkDescriptorSetLayoutBinding contributions_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
    VkDescriptorSetLayoutBinding sorted_keys_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
    VkDescriptorSetLayoutBinding u_keys_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
	VkDescriptorSetLayoutBinding reduction_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding index_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);


	std::vector<VkDescriptorSetLayoutBinding> find_dup_set_layout_bindings = {
		sorted_keys_layoutBinding, contributions_layoutBinding
	};

	std::vector<VkDescriptorSetLayoutBinding> prefix_sum_set_layout_bindings = {
		index_layoutBinding, contributions_layoutBinding, reduction_layoutBinding
	};

	std::vector<VkDescriptorSetLayoutBinding> move_dup_set_layout_bindings = {
		sorted_keys_layoutBinding, contributions_layoutBinding, u_keys_layoutBinding
	};
	// create descriptor 
	create_descriptor_set_layout(find_dup_set_layout_bindings, &descriptorLayout[1], &descriptorSetLayouts[1]);
	create_descriptor_set_layout(prefix_sum_set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);
	create_descriptor_set_layout(move_dup_set_layout_bindings, &descriptorLayout[2], &descriptorSetLayouts[2]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(3, descriptorSetLayouts);
	//add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(3, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
	// find_dup
	VkDescriptorBufferInfo contributions_bufferDescriptor = { buffer.contributions_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet contributions_descriptor_write = create_descriptor_write(descriptorSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &contributions_bufferDescriptor);
	VkDescriptorBufferInfo sorted_keys_bufferDescriptor = { buffer.sorted_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet sorted_keys_descriptor_write = create_descriptor_write(descriptorSets[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &sorted_keys_bufferDescriptor);

	// prefix sum
    VkDescriptorBufferInfo contributions_bufferDescriptor_2 = { buffer.contributions_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet contributions_descriptor_write_2  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &contributions_bufferDescriptor_2);
    VkDescriptorBufferInfo reduction_bufferDescriptor = { buffer.reduction_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet reduction_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &reduction_bufferDescriptor);
    VkDescriptorBufferInfo index_bufferDescriptor = { buffer.index_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet index_descriptor_write = create_descriptor_write(descriptorSets[0],2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &index_bufferDescriptor);

	// move dup
	VkDescriptorBufferInfo contributions_bufferDescriptor_3 = { buffer.contributions_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet contributions_descriptor_write_3 = create_descriptor_write(descriptorSets[2], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &contributions_bufferDescriptor_3);
	VkDescriptorBufferInfo sorted_keys_bufferDescriptor_2 = { buffer.sorted_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet sorted_keys_descriptor_write_2 = create_descriptor_write(descriptorSets[2], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &sorted_keys_bufferDescriptor_2);
	VkDescriptorBufferInfo u_keys_bufferDescriptor = { buffer.u_keys_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet u_keys_descriptor_write = create_descriptor_write(descriptorSets[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &u_keys_bufferDescriptor);


	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
		contributions_descriptor_write, sorted_keys_descriptor_write, contributions_descriptor_write_2, reduction_descriptor_write, index_descriptor_write, contributions_descriptor_write_3, sorted_keys_descriptor_write_2, u_keys_descriptor_write
	};


	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);

	//create pipeline 
	VkPipelineShaderStageCreateInfo find_dup_shader_stage = load_shader("find_dups.spv", &find_dups_shaderModule);
	create_pipeline(&find_dup_shader_stage,&pipelineLayout, &find_dup_pipeline);
	VkPipelineShaderStageCreateInfo prefix_sum_shader_stage = load_shader("prefix_sum.spv", &prefix_sum_shaderModule);
	create_pipeline(&prefix_sum_shader_stage,&pipelineLayout, &prefix_sum_pipeline);
	VkPipelineShaderStageCreateInfo move_dup_shader_stage = load_shader("move_dups.spv", &move_dups_shaderModule);
	create_pipeline(&move_dup_shader_stage,&pipelineLayout, &move_dup_pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
    VkBufferMemoryBarrier sorted_keys_barrier = create_buffer_barrier(&buffer.sorted_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier contributions_barrier = create_buffer_barrier(&buffer.contributions_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier u_keys_barrier = create_buffer_barrier(&buffer.u_keys_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier reduction_barrier = create_buffer_barrier(&buffer.reduction_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier index_barrier = create_buffer_barrier(&buffer.index_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);



    create_pipeline_barrier(&sorted_keys_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&contributions_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&u_keys_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&reduction_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&index_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for move_dup
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &unique_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, move_dup_pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 3, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_block, 1, 1);

    contributions_barrier = create_buffer_barrier(&buffer.u_keys_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    create_pipeline_barrier(&contributions_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	
	// for prefix_sum
	unique_push_constant.n = aligned_size;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &unique_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_sum_pipeline);
	vkCmdDispatch(commandBuffer, num_blocks, 1, 1);

	contributions_barrier = create_buffer_barrier(&buffer.contributions_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&contributions_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	

	// for find_dup
	unique_push_constant.n = n;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &unique_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, find_dup_pipeline);
	vkCmdDispatch(commandBuffer, logical_block, 1, 1);

	u_keys_barrier = create_buffer_barrier(&buffer.u_keys_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&u_keys_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferCopy copyRegion = {};
	copyRegion.size = n* sizeof(uint32_t);
	vkCmdCopyBuffer(commandBuffer, buffer.u_keys_buffer, temp_buffer.u_keys_buffer, 1, &copyRegion);
	u_keys_barrier = create_buffer_barrier(&buffer.u_keys_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&u_keys_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

	VkBufferCopy prefix_sum_copyRegion = {};
	prefix_sum_copyRegion.size = n* sizeof(uint32_t);
	vkCmdCopyBuffer(commandBuffer, buffer.contributions_buffer, temp_buffer.contributions_buffer, 1, &prefix_sum_copyRegion);
	contributions_barrier = create_buffer_barrier(&buffer.contributions_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&contributions_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	execute();

	// Make device writes visible to the host
	void *mapped;
	vkMapMemory(singleton.device, temp_memory.u_keys_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = temp_memory.u_keys_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
			
	const VkDeviceSize bufferSize = n * sizeof(uint32_t);
	memcpy(u_keys, mapped, bufferSize);
	vkUnmapMemory(singleton.device, temp_memory.u_keys_memory);

	vkMapMemory(singleton.device, temp_memory.contributions_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	mappedRange.memory = temp_memory.contributions_memory;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
	memcpy(contributions, mapped, bufferSize);
	vkUnmapMemory(singleton.device, temp_memory.contributions_memory);


	vkQueueWaitIdle(singleton.queue);


	cleanup(&find_dup_pipeline, &prefix_sum_pipeline, &move_dup_pipeline);
}

