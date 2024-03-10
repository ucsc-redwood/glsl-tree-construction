#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define PARTITION_SIZE 7680
#define BINNING_THREAD_BLOCKS  (n + PARTITION_SIZE - 1) / PARTITION_SIZE

class RadixSort : public ApplicationBase{
    public:
    RadixSort() : ApplicationBase() {};
    ~RadixSort() {};
	void submit();
	void 		 cleanup(VkPipeline *histogram_pipeline, VkPipeline *binning_pipeline);
	void run(const int logical_blocks, uint32_t* computeInput, const int n);

    private:
	VkShaderModule histogram_shaderModule;
	VkShaderModule binning_shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[2] = {VkDescriptorSetLayout{}, VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[2] = {VkDescriptorSet{}, VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[2] = {VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}};
	struct RadixSortPushConstant {
		uint32_t pass_num = 0;
		uint32_t radix_shift = 0;
	} radix_sort_push_constant;
	struct{
		VkBuffer b_sort_buffer;
		VkBuffer g_histogram_buffer;
		VkBuffer b_alt_buffer;
		VkBuffer b_index_buffer;
		VkBuffer b_pass_first_histogram_buffer;
		VkBuffer pass_num_buffer;
	} radix_sort_buffer;

	struct{
		VkBuffer b_sort_buffer;
		VkBuffer g_histogram_buffer;
		VkBuffer b_alt_buffer;
		VkBuffer b_index_buffer;
		VkBuffer b_pass_first_histogram_buffer;
		VkBuffer pass_num_buffer;
		
	} temp_buffer;

	struct{
		VkDeviceMemory b_sort_memory;
		VkDeviceMemory g_histogram_memory;
		VkDeviceMemory b_alt_memory;
		VkDeviceMemory b_index_memory;
		VkDeviceMemory b_pass_first_histogram_memory;
		VkDeviceMemory pass_num_memory;
	} radix_sort_memory;

	struct{
		VkDeviceMemory b_sort_memory;
		VkDeviceMemory g_histogram_memory;
		VkDeviceMemory b_alt_memory;
		VkDeviceMemory b_index_memory;
		VkDeviceMemory b_pass_first_histogram_memory;
		VkDeviceMemory pass_num_memory;
	} temp_memory;

};


void RadixSort::submit(){
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

void RadixSort::cleanup(VkPipeline *histogram_pipeline, VkPipeline *binning_pipeline){
		vkDestroyBuffer(singleton.device, radix_sort_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(singleton.device, radix_sort_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(singleton.device, radix_sort_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(singleton.device, radix_sort_memory.g_histogram_memory, nullptr);

		vkDestroyBuffer(singleton.device, temp_buffer.b_sort_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.b_sort_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.g_histogram_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.g_histogram_memory, nullptr);

		vkDestroyBuffer(singleton.device, radix_sort_buffer.b_alt_buffer, nullptr);
		vkFreeMemory(singleton.device, radix_sort_memory.b_alt_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.b_alt_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.b_alt_memory, nullptr);

		vkDestroyBuffer(singleton.device, radix_sort_buffer.b_index_buffer, nullptr);
		vkFreeMemory(singleton.device, radix_sort_memory.b_index_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.b_index_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.b_index_memory, nullptr);
		
		vkDestroyBuffer(singleton.device, radix_sort_buffer.b_pass_first_histogram_buffer, nullptr);
		vkFreeMemory(singleton.device, radix_sort_memory.b_pass_first_histogram_memory, nullptr);
		vkDestroyBuffer(singleton.device, temp_buffer.b_pass_first_histogram_buffer, nullptr);
		vkFreeMemory(singleton.device, temp_memory.b_pass_first_histogram_memory, nullptr);


		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[1], nullptr);
		vkDestroyPipeline(singleton.device, *histogram_pipeline, nullptr);
		vkDestroyPipeline(singleton.device, *binning_pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, binning_shaderModule, nullptr);
		vkDestroyShaderModule(singleton.device, histogram_shaderModule, nullptr);
		
}

void RadixSort::run(const int logical_blocks, uint32_t* computeInput, const int n){
	 
	std::vector<uint32_t> g_histogram(1024, 0);
	std::vector<uint32_t> b_alt(n, 0);
	std::vector<uint32_t> b_index(4, 0);
	std::vector<glm::uvec4> b_pass_first_histogram(BINNING_THREAD_BLOCKS*1024, glm::uvec4(0, 0, 0, 0));
	
	VkPipeline histogram_pipeline;
	VkPipeline binning_pipeline;

	const VkDeviceSize bufferSize = n * sizeof(uint32_t);
	/*
	create_instance();
	create_device();
	create_compute_queue();
	build_command_pool();
	*/
	create_storage_buffer(bufferSize, computeInput, &radix_sort_buffer.b_sort_buffer, &radix_sort_memory.b_sort_memory, &temp_buffer.b_sort_buffer, &temp_memory.b_sort_memory);
	create_storage_buffer(1024*sizeof(uint32_t), g_histogram.data(), &radix_sort_buffer.g_histogram_buffer, &radix_sort_memory.g_histogram_memory, &temp_buffer.g_histogram_buffer, &temp_memory.g_histogram_memory);
	create_storage_buffer(bufferSize, b_alt.data(), &radix_sort_buffer.b_alt_buffer, &radix_sort_memory.b_alt_memory, &temp_buffer.b_alt_buffer, &temp_memory.b_alt_memory);
	create_storage_buffer(4*sizeof(uint32_t), b_index.data(), &radix_sort_buffer.b_index_buffer, &radix_sort_memory.b_index_memory, &temp_buffer.b_index_buffer, &temp_memory.b_index_memory);
	create_storage_buffer(b_pass_first_histogram.size()*sizeof(glm::uvec4), b_pass_first_histogram.data(), &radix_sort_buffer.b_pass_first_histogram_buffer, &radix_sort_memory.b_pass_first_histogram_memory, &temp_buffer.b_pass_first_histogram_buffer, &temp_memory.b_pass_first_histogram_memory);

	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9},
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
	};

	create_descriptor_pool(poolSizes, 2);

	// create layout binding
	VkDescriptorSetLayoutBinding b_sort_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_alt_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
	VkDescriptorSetLayoutBinding b_global_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding b_index_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);
	VkDescriptorSetLayoutBinding b_pass_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1);
	std::vector<VkDescriptorSetLayoutBinding> histogram_set_layout_bindings = {
		b_sort_layoutBinding, b_global_hist_layoutBinding
	};
	std::vector<VkDescriptorSetLayoutBinding> binning_set_layout_bindings = {
		b_sort_layoutBinding, b_alt_layoutBinding, b_global_hist_layoutBinding, b_index_layoutBinding, b_pass_hist_layoutBinding
	};

	// create descriptor set layout for both histogram and binning
	create_descriptor_set_layout(histogram_set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);
	create_descriptor_set_layout(binning_set_layout_bindings, &descriptorLayout[1], &descriptorSetLayouts[1]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(2, descriptorSetLayouts);
	//add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(2, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
	// for histogram
	VkDescriptorBufferInfo b_sort_bufferDescriptor = { radix_sort_buffer.b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_bufferDescriptor = { radix_sort_buffer.g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_bufferDescriptor);
	
	// for binning
	VkDescriptorBufferInfo b_sort_binning_bufferDescriptor = { radix_sort_buffer.b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_binning_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_binning_bufferDescriptor = { radix_sort_buffer.g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_binning_descriptor_write = create_descriptor_write(descriptorSets[1],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_binning_bufferDescriptor);
	VkDescriptorBufferInfo b_alt_binning_bufferDescriptor = { radix_sort_buffer.b_alt_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_alt_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_alt_binning_bufferDescriptor);
	VkDescriptorBufferInfo b_index_binning_bufferDescriptor = { radix_sort_buffer.b_index_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_index_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_index_binning_bufferDescriptor);
	
	VkDescriptorBufferInfo b_pass_first_histogram_binning_bufferDescriptor = { radix_sort_buffer.b_pass_first_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_pass_histogram_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_pass_first_histogram_binning_bufferDescriptor);
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {b_sort_descriptor_write, g_histogram_descriptor_write,  b_sort_binning_descriptor_write, g_histogram_binning_descriptor_write, b_alt_binning_descriptor_write, b_index_binning_descriptor_write, b_pass_histogram_binning_descriptor_write};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline for histogram
	VkPipelineShaderStageCreateInfo histogram_shader_stage = load_shader("histogram.spv", &histogram_shaderModule);
	create_pipeline(&histogram_shader_stage,&pipelineLayout, &histogram_pipeline);

	//create pipeline for binning
	VkPipelineShaderStageCreateInfo binning_shader_stage = load_shader("new_radix_sort.spv", &binning_shaderModule);
	create_pipeline(&binning_shader_stage,&pipelineLayout, &binning_pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_alt_barrier = create_buffer_barrier(&radix_sort_buffer.b_alt_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_index_barrier = create_buffer_barrier(&radix_sort_buffer.b_index_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_pass_first_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.b_pass_first_histogram_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_first_histogram_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for histogram
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogram_pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 2, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	
	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	

	// for first binning
		// push data to the push constants
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, binning_pipeline);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);

	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&radix_sort_buffer.b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&radix_sort_buffer.b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_first_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.b_pass_first_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_first_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	

	// for second binning
	radix_sort_push_constant.pass_num = 1;
	radix_sort_push_constant.radix_shift = 8;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&radix_sort_buffer.b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&radix_sort_buffer.b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_first_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.b_pass_first_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_first_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for third binning
	radix_sort_push_constant.pass_num = 2;
	radix_sort_push_constant.radix_shift = 16;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&radix_sort_buffer.b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&radix_sort_buffer.b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_first_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.b_pass_first_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_first_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for fourth binning
	radix_sort_push_constant.pass_num = 3;
	radix_sort_push_constant.radix_shift = 24;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&radix_sort_buffer.b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&radix_sort_buffer.b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_pass_first_histogram_barrier = create_buffer_barrier(&radix_sort_buffer.b_pass_first_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_pass_first_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

	VkBufferCopy copyRegion = {};
	copyRegion.size = 1024* sizeof(uint32_t);
	vkCmdCopyBuffer(commandBuffer, radix_sort_buffer.b_sort_buffer, temp_buffer.b_sort_buffer, 1, &copyRegion);
	b_sort_barrier = create_buffer_barrier(&radix_sort_buffer.b_sort_buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);


	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit();

	void *mapped;
	vkMapMemory(singleton.device, temp_memory.b_sort_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
	VkMappedMemoryRange mappedRange{};
	mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
	mappedRange.memory = /*temp_memory.g_histogram_memory;*/temp_memory.b_sort_memory;
	mappedRange.offset = 0;
	mappedRange.size = VK_WHOLE_SIZE;
	vkInvalidateMappedMemoryRanges(singleton.device, 1, &mappedRange);
			
	// todo: change the buffer size
	// Copy to output
	memcpy(computeInput, mapped, bufferSize);
	vkUnmapMemory(singleton.device, temp_memory.b_sort_memory);

	vkQueueWaitIdle(singleton.queue);

	// Make device writes visible to the host


	cleanup(&histogram_pipeline, &binning_pipeline);
}

