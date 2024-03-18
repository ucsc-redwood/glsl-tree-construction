#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>


struct OctNode{
    int children[8];
    glm::vec4 corner;
    float cell_size;
    int child_node_mask;
    int child_leaf_mask;

    explicit OctNode(){
        for (int i = 0; i < 8; ++i){
            children[i] = -1;
        }
        corner = glm::vec4(0,0,0,0);
        cell_size = 0;
        child_node_mask = 0;
        child_leaf_mask = 0;
    }
};

class Octree : public ApplicationBase{
    public:
    Octree() : ApplicationBase() {};
    ~Octree() {};
	void 		submit(const int queue_idx);
	void 		cleanup(VkPipeline *pipeline);
	void 		run(
    const int logical_blocks,
    const int queue_idx,
    // --- output parameters
    OctNode* oct_nodes,
    // --- end output parameters, begin input parameters (read-only)
    uint32_t* node_offsets,
    int* node_counts,
    unsigned int* codes,
    uint8_t* rt_prefixN,
    bool* rt_hasLeafLeft,
    bool* rt_hasLeafRight,
    int* rt_parents,
    int* rt_leftChild,
    float min_coord,
    float range,
    int n_brt_nodes,
    VkBuffer oct_node_buffer,
    VkBuffer node_offsets_buffer,
    VkBuffer node_counts_buffer,
    VkBuffer codes_buffer,
    VkBuffer rt_prefixN_buffer,
    VkBuffer rt_hasLeafLeft_buffer,
    VkBuffer rt_hasLeafRight_buffer,
    VkBuffer rt_parents_buffer,
    VkBuffer rt_leftChild_buffer 
    );

    private:
	VkShaderModule shaderModule;


	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};

	struct PushConstant {
        float min_coord;
        float range;
        int n_brt_nodes;
	} octree_push_constant;
    

};


void Octree::submit(const int queue_idx){
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

void Octree::cleanup(VkPipeline *pipeline){

	vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
	vkDestroyPipeline(singleton.device, *pipeline, nullptr);
	vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void Octree::run(
    const int logical_blocks,
    const int queue_idx,
    // --- output parameters
    OctNode* oct_nodes,
    // --- end output parameters, begin input parameters (read-only)
    uint32_t* node_offsets,
    int* node_counts,
    unsigned int* codes,
    uint8_t* rt_prefixN,
    bool* rt_hasLeafLeft,
    bool* rt_hasLeafRight,
    int* rt_parents,
    int* rt_leftChild,
    float min_coord,
    float range,
    int n_brt_nodes,
    VkBuffer oct_node_buffer,
    VkBuffer node_offsets_buffer,
    VkBuffer node_counts_buffer,
    VkBuffer codes_buffer,
    VkBuffer rt_prefixN_buffer,
    VkBuffer rt_hasLeafLeft_buffer,
    VkBuffer rt_hasLeafRight_buffer,
    VkBuffer rt_parents_buffer,
    VkBuffer rt_leftChild_buffer){
	 
	
	VkPipeline pipeline;


	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
    VkDescriptorSetLayoutBinding oct_node_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
    VkDescriptorSetLayoutBinding node_offsets_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
    VkDescriptorSetLayoutBinding node_counts_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
    VkDescriptorSetLayoutBinding codes_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);
    VkDescriptorSetLayoutBinding rt_prefixN_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4, 1);
    VkDescriptorSetLayoutBinding rt_hasLeafLeft_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5, 1);
    VkDescriptorSetLayoutBinding rt_hasLeafRight_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6, 1);
    VkDescriptorSetLayoutBinding rt_parents_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7, 1);
    VkDescriptorSetLayoutBinding rt_leftChild_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8, 1);
    
	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
        oct_node_layoutBinding, node_offsets_layoutBinding, node_counts_layoutBinding, codes_layoutBinding, rt_prefixN_layoutBinding, rt_hasLeafLeft_layoutBinding, rt_hasLeafRight_layoutBinding, rt_parents_layoutBinding, rt_leftChild_layoutBinding
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

    VkDescriptorBufferInfo oct_node_bufferDescriptor = { oct_node_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet oct_node_descriptor_write = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &oct_node_bufferDescriptor);
    VkDescriptorBufferInfo node_offsets_bufferDescriptor = { node_offsets_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet node_offsets_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &node_offsets_bufferDescriptor);
    VkDescriptorBufferInfo node_counts_bufferDescriptor = { node_counts_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet node_counts_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &node_counts_bufferDescriptor);
    VkDescriptorBufferInfo codes_bufferDescriptor = { codes_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet codes_descriptor_write = create_descriptor_write(descriptorSets[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &codes_bufferDescriptor);
    VkDescriptorBufferInfo rt_prefixN_bufferDescriptor = { rt_prefixN_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet rt_prefixN_descriptor_write = create_descriptor_write(descriptorSets[0], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &rt_prefixN_bufferDescriptor);
    VkDescriptorBufferInfo rt_hasLeafLeft_bufferDescriptor = { rt_hasLeafLeft_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet rt_hasLeafLeft_descriptor_write = create_descriptor_write(descriptorSets[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &rt_hasLeafLeft_bufferDescriptor);
    VkDescriptorBufferInfo rt_hasLeafRight_bufferDescriptor = { rt_hasLeafRight_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet rt_hasLeafRight_descriptor_write = create_descriptor_write(descriptorSets[0], 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &rt_hasLeafRight_bufferDescriptor);
    VkDescriptorBufferInfo rt_parents_bufferDescriptor = { rt_parents_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet rt_parents_descriptor_write = create_descriptor_write(descriptorSets[0], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &rt_parents_bufferDescriptor);
    VkDescriptorBufferInfo rt_leftChild_bufferDescriptor = { rt_leftChild_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet rt_leftChild_descriptor_write = create_descriptor_write(descriptorSets[0], 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &rt_leftChild_bufferDescriptor);
	
	
	std::vector<VkWriteDescriptorSet> descriptor_writes = {
        oct_node_descriptor_write, node_offsets_descriptor_write, node_counts_descriptor_write, codes_descriptor_write, rt_prefixN_descriptor_write, rt_hasLeafLeft_descriptor_write, rt_hasLeafRight_descriptor_write, rt_parents_descriptor_write, rt_leftChild_descriptor_write
	};
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	
	// create pipeline
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("octree.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);



	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
    VkBufferMemoryBarrier oct_node_barrier = create_buffer_barrier(&oct_node_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier node_offsets_barrier = create_buffer_barrier(&node_offsets_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier node_counts_barrier = create_buffer_barrier(&node_counts_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier codes_barrier = create_buffer_barrier(&codes_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier rt_prefixN_barrier = create_buffer_barrier(&rt_prefixN_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier rt_hasLeafLeft_barrier = create_buffer_barrier(&rt_hasLeafLeft_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier rt_hasLeafRight_barrier = create_buffer_barrier(&rt_hasLeafRight_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier rt_parents_barrier = create_buffer_barrier(&rt_parents_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier rt_leftChild_barrier = create_buffer_barrier(&rt_leftChild_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    create_pipeline_barrier(&oct_node_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&node_offsets_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&node_counts_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&codes_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&rt_prefixN_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&rt_hasLeafLeft_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&rt_hasLeafRight_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&rt_parents_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&rt_leftChild_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);



	octree_push_constant.n_brt_nodes = n_brt_nodes;
    octree_push_constant.min_coord = min_coord;
    octree_push_constant.range = range;
	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &octree_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, logical_blocks, 1, 1);
	
    node_offsets_barrier = create_buffer_barrier(&node_offsets_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    node_counts_barrier = create_buffer_barrier(&node_counts_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    codes_barrier = create_buffer_barrier(&codes_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt_prefixN_barrier = create_buffer_barrier(&rt_prefixN_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt_hasLeafLeft_barrier = create_buffer_barrier(&rt_hasLeafLeft_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt_hasLeafRight_barrier = create_buffer_barrier(&rt_hasLeafRight_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt_parents_barrier = create_buffer_barrier(&rt_parents_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt_leftChild_barrier = create_buffer_barrier(&rt_leftChild_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    create_pipeline_barrier(&node_offsets_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&node_counts_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&codes_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&rt_prefixN_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&rt_hasLeafLeft_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&rt_hasLeafRight_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&rt_parents_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    create_pipeline_barrier(&rt_leftChild_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


	vkEndCommandBuffer(commandBuffer);

	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit(queue_idx);


	vkQueueWaitIdle(singleton.queues[queue_idx]);

	cleanup(&pipeline);
}

