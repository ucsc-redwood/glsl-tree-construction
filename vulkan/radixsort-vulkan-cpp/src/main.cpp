#include "vma_usage.hpp"
#include "radixsort.hpp"
#include <vulkan/vulkan.hpp>

int main(){
    auto app = RadixSort();
    app.create_instance();
}