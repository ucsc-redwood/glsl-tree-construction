#include "app_params.hpp"
#include "init.hpp"
#include "morton.hpp"
#include "unique.hpp"
#include "radixsort.hpp"
#include "radix_tree.hpp"
#include "edge_count.hpp"
#include "prefix_sum.hpp"
#include "octree.hpp"
#include <vulkan/vulkan.hpp>

#define BUFFER_ELEMENTS 131072
#define BLOCK_NUM 16
int main(){
    AppParams app_params;
    app_params.n = BUFFER_ELEMENTS;
    app_params.min_coord = 0.0f;
    app_params.max_coord = 1.0f;
    app_params.seed = 114514;
    app_params.n_threads = 4;
    app_params.n_blocks = 16;

    std::vector<glm::vec4> data(BUFFER_ELEMENTS ,glm::vec4(0,0,0,0));
    Init init_stage = Init();
    init_stage.run(BLOCK_NUM, data.data(), app_params.n, app_params.min_coord, app_params.getRange(), app_params.seed);
    for (int i = 0; i < 1024; ++i){
        std::cout << data[i].x << " " << data[i].y << " " << data[i].z << " " << data[i].w << std::endl;
    }
    /*
    Morton morton_stage = Morton();
    morton_stage.run();
    auto radixsort = RadixSort();
    radixsort.run();
    */
}