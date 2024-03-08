#include "radixsort.hpp"
#include <vulkan/vulkan.hpp>

int main(){
    auto app = RadixSort();
    //auto result = app.create_instance();
    //printf("Create instance result: %d\n", result);
    app.run();
}