#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>
#include <chrono>

#define SORT_PARTITION_SIZE 3840
#define RADIX_BIN 256
#define THREAD_BLOCKS  (n + SORT_PARTITION_SIZE - 1) / SORT_PARTITION_SIZE

class RadixSort64 : public ApplicationBase{
    public:
    RadixSort64() : ApplicationBase() {};
    ~RadixSort64() {};
	void submit(const int queue_idx);
	void 		 cleanup(VkPipeline *histogram_pipeline, VkPipeline * scan_pipeline, VkPipeline *binning_pipeline);
	void run(
	const int logical_blocks,
	const int queue_idx,
	uint32_t* b_sort, 
	uint32_t* b_alt,
	uint32_t* g_histogram,
	uint32_t* b_index,
	uint32_t* b_pass_histogram,
	VkBuffer b_sort_buffer,
	VkBuffer b_alt_buffer,
	VkBuffer g_histogram_buffer,
	VkBuffer b_index_buffer,
	VkBuffer b_pass_histogram_buffer,
	const int n
	);

    private:
	VkShaderModule histogram_shaderModule;
    VkShaderModule scan_shaderModule;
	VkShaderModule binning_shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[3] = {VkDescriptorSetLayout{}, VkDescriptorSetLayout{}, VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[3] = {VkDescriptorSet{}, VkDescriptorSet{}, VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[3] = {VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}, VkDescriptorSetLayoutCreateInfo{}};
	struct RadixSortPushConstant {
		uint32_t pass_num = 0;
		uint32_t radix_shift = 0;
		uint32_t n = 0;
        uint32_t e_workgroups = 0;

	} radix_sort_push_constant;
};


void RadixSort64::submit(const int queue_idx){
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

void RadixSort64::cleanup(VkPipeline *histogram_pipeline,  VkPipeline * scan_pipeline,VkPipeline *binning_pipeline){

		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[1], nullptr);
        vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[2], nullptr);
        vkDestroyPipelineLayout(singleton.device, pipelineLayout, nullptr);
		vkDestroyPipeline(singleton.device, *histogram_pipeline, nullptr);
        vkDestroyPipeline(singleton.device, *scan_pipeline, nullptr);
		vkDestroyPipeline(singleton.device, *binning_pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, binning_shaderModule, nullptr);
        vkDestroyShaderModule(singleton.device, scan_shaderModule, nullptr);
		vkDestroyShaderModule(singleton.device, histogram_shaderModule, nullptr);
		
}

void RadixSort64::run(const int logical_blocks, 
	const int queue_idx,
	uint32_t* b_sort, 
	uint32_t* b_alt,
	uint32_t* g_histogram,
	uint32_t* b_index,
	uint32_t* b_pass_histogram,
	VkBuffer b_sort_buffer,
	VkBuffer b_alt_buffer,
	VkBuffer g_histogram_buffer,
	VkBuffer b_index_buffer,
	VkBuffer b_pass_histogram_buffer,
	const int n){
	 	
	VkPipeline histogram_pipeline;
    VkPipeline scan_pipeline;
	VkPipeline binning_pipeline;


	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9},
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
	};

	create_descriptor_pool(poolSizes, 3);

	// create layout binding
	VkDescriptorSetLayoutBinding b_sort_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
	VkDescriptorSetLayoutBinding b_alt_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
	VkDescriptorSetLayoutBinding b_global_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
	VkDescriptorSetLayoutBinding b_index_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);
	VkDescriptorSetLayoutBinding b_pass_hist_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1);
	std::vector<VkDescriptorSetLayoutBinding> histogram_set_layout_bindings = {
		b_sort_layoutBinding, b_global_hist_layoutBinding
	};
    std::vector <VkDescriptorSetLayoutBinding> scan_set_layout_bindings = {
        b_global_hist_layoutBinding, b_pass_hist_layoutBinding
    };
	std::vector<VkDescriptorSetLayoutBinding> binning_set_layout_bindings = {
		b_sort_layoutBinding, b_alt_layoutBinding, b_global_hist_layoutBinding, b_index_layoutBinding, b_pass_hist_layoutBinding
	};

	// create descriptor set layout for both histogram and binning
	create_descriptor_set_layout(histogram_set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);
	create_descriptor_set_layout(binning_set_layout_bindings, &descriptorLayout[1], &descriptorSetLayouts[1]);
    create_descriptor_set_layout(scan_set_layout_bindings, &descriptorLayout[2], &descriptorSetLayouts[2]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(3, descriptorSetLayouts);
	//add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);

	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(3, descriptorSetLayouts, descriptorSets);

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
	// for histogram
	VkDescriptorBufferInfo b_sort_bufferDescriptor = { b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_bufferDescriptor = { g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_descriptor_write = create_descriptor_write(descriptorSets[0],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_bufferDescriptor);

    // for scan
    VkDescriptorBufferInfo g_histogram_scan_bufferDescriptor = { g_histogram_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet g_histogram_scan_descriptor_write = create_descriptor_write(descriptorSets[2], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_scan_bufferDescriptor);
    VkDescriptorBufferInfo b_pass_histogram_scan_bufferDescriptor = { b_pass_histogram_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet b_pass_histogram_scan_descriptor_write = create_descriptor_write(descriptorSets[2], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_pass_histogram_scan_bufferDescriptor);
	
	// for binning
	VkDescriptorBufferInfo b_sort_binning_bufferDescriptor = { b_sort_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_sort_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_sort_binning_bufferDescriptor);
	VkDescriptorBufferInfo g_histogram_binning_bufferDescriptor = { g_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet g_histogram_binning_descriptor_write = create_descriptor_write(descriptorSets[1],1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &g_histogram_binning_bufferDescriptor);
	VkDescriptorBufferInfo b_alt_binning_bufferDescriptor = { b_alt_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_alt_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_alt_binning_bufferDescriptor);
	VkDescriptorBufferInfo b_index_binning_bufferDescriptor = { b_index_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_index_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_index_binning_bufferDescriptor);

	VkDescriptorBufferInfo b_pass_histogram_binning_bufferDescriptor = { b_pass_histogram_buffer, 0, VK_WHOLE_SIZE };
	VkWriteDescriptorSet b_pass_histogram_binning_descriptor_write  = create_descriptor_write(descriptorSets[1], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &b_pass_histogram_binning_bufferDescriptor);
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {b_sort_descriptor_write,
    g_histogram_descriptor_write,
    g_histogram_scan_descriptor_write,
    b_pass_histogram_scan_descriptor_write,
    b_sort_binning_descriptor_write,
    g_histogram_binning_descriptor_write,
    b_alt_binning_descriptor_write,
    b_index_binning_descriptor_write,
    b_pass_histogram_binning_descriptor_write};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline for histogram
	VkPipelineShaderStageCreateInfo histogram_shader_stage = load_shader("histogram_64.spv", &histogram_shaderModule);
	create_pipeline(&histogram_shader_stage,&pipelineLayout, &histogram_pipeline);
    
    // create pipeline for scan
    VkPipelineShaderStageCreateInfo scan_shader_stage = load_shader("radix_sort_scan.spv", &scan_shaderModule);
    create_pipeline(&scan_shader_stage, &pipelineLayout, &scan_pipeline);
	//create pipeline for binning
	VkPipelineShaderStageCreateInfo binning_shader_stage = load_shader("radix_sort_64.spv", &binning_shaderModule);
	create_pipeline(&binning_shader_stage,&pipelineLayout, &binning_pipeline);

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
	VkBufferMemoryBarrier b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_alt_barrier = create_buffer_barrier(&b_alt_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_index_barrier = create_buffer_barrier(&b_index_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VkBufferMemoryBarrier b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for histogram
	radix_sort_push_constant.n = n;
    radix_sort_push_constant.e_workgroups = THREAD_BLOCKS; 
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogram_pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 3, descriptorSets, 0, 0);
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, THREAD_BLOCKS, 1, 1);
	
	b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	
    // for scan
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scan_pipeline);
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
    vkCmdDispatch(commandBuffer, RADIX_BIN, 1, 1);
    g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	// for first binning
		// push data to the push constants
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, binning_pipeline);
		radix_sort_push_constant.pass_num = 0;
	radix_sort_push_constant.radix_shift = 0;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, THREAD_BLOCKS, 1, 1);

	b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	
	
	// for second binning
	radix_sort_push_constant.pass_num = 1;
	radix_sort_push_constant.radix_shift = 8;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, THREAD_BLOCKS, 1, 1);
	b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	
	// for third binning
	radix_sort_push_constant.pass_num = 2;
	radix_sort_push_constant.radix_shift = 16;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, THREAD_BLOCKS, 1, 1);
	b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// for fourth binning
	radix_sort_push_constant.pass_num = 3;
	radix_sort_push_constant.radix_shift = 24;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RadixSortPushConstant), &radix_sort_push_constant);
	vkCmdDispatch(commandBuffer, THREAD_BLOCKS, 1, 1);
	b_sort_barrier = create_buffer_barrier(&b_sort_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_alt_barrier = create_buffer_barrier(&b_alt_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	g_histogram_barrier = create_buffer_barrier(&g_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_index_barrier = create_buffer_barrier(&b_index_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	b_pass_histogram_barrier = create_buffer_barrier(&b_pass_histogram_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
	create_pipeline_barrier(&b_sort_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_alt_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&g_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_index_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	create_pipeline_barrier(&b_pass_histogram_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	
	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	const auto start = std::chrono::high_resolution_clock::now();
	// submit the command buffer, fence and flush
	submit(queue_idx);
	const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "time: " << elapsed.count() << "ms" << std::endl;

	vkQueueWaitIdle(singleton.queues[queue_idx]);

	// Make device writes visible to the host


	cleanup(&histogram_pipeline, &scan_pipeline,  &binning_pipeline);
}

