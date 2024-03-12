#pragma once
#include "application.hpp"
#include <glm/glm.hpp>
#include <iostream>

#define PARTITION_SIZE 7680
#define BINNING_THREAD_BLOCKS  (n + PARTITION_SIZE - 1) / PARTITION_SIZE

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
		VkDeviceMemory data_memory;
	} memory;

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
			vkQueueSubmit(singleton.queue, 1, &computeSubmitInfo, fence);
			vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);

}

void Init::cleanup(VkPipeline *pipeline){
		vkUnmapMemory(singleton.device, memory.data_memory);

		vkDestroyBuffer(singleton.device, buffer.data_buffer, nullptr);
		vkFreeMemory(singleton.device, memory.data_memory, nullptr);
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
		
}

void Init::run(const int blocks, glm::vec4* data, const int n, const int min_val, const float range, const float seed){
	 std::cout <<"init"<<std::endl;
	
	VkPipeline pipeline;


	glm::vec4* test_data = reinterpret_cast<glm::vec4*>(create_shared_storage_buffer(n*sizeof(glm::vec4), &buffer.data_buffer, &memory.data_memory));
	
	memcpy(test_data, data, sizeof(glm::vec4));
}

