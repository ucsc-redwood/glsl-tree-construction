#include "morton.hpp"
#include "unique.hpp"
#include "radixsort.hpp"
#include "radix_tree.hpp"
#include "edge_count.hpp"
#include "prefix_sum.hpp"
#include "octree.hpp"
#include <vulkan/vulkan.hpp>

int main(){
    auto app = RadixSort();
    //auto result = app.create_instance();
    //printf("Create instance result: %d\n", result);
    app.run();
}