#include "vma_usage.hpp"
#include "radixsort.hpp"
#include <vulkan/vulkan.hpp>

int main(){
    auto app = RadixSort();
    auto result = app.create_instance();
    printf("Create instance result: %d\n", result);
    app.create_device();
    app.create_compute_queue();
    app.build_compute_pipeline();
    app.build_command_pool();
    app.create_storage_buffer();
    app.create_command_buffer();
}