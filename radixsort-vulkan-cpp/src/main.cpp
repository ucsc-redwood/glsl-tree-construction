#include "app_params.hpp"
#include "naive_pipe.hpp"
#include <vulkan/vulkan.hpp>
#include <chrono>

#define BUFFER_ELEMENTS 1920 * 1080

int main(const int argc, const char *argv[])
{
    int n_blocks = 2;

    if (argc > 1)
    {
        n_blocks = std::stoi(argv[1]);
    }
    AppParams app_params;
    app_params.n = BUFFER_ELEMENTS;
    app_params.min_coord = 0.0f;
    app_params.max_coord = 1.0f;
    app_params.seed = 114514;
    app_params.n_threads = 4;
    app_params.n_blocks = n_blocks;

    Pipe pipe = Pipe(app_params);
    pipe.allocate();

    std::cout << "num blocks: " << n_blocks << std::endl;
    pipe.init(n_blocks, 0);

    pipe.morton(n_blocks, 0);

    pipe.radix_sort(n_blocks, 0);

    pipe.unique(n_blocks, 0);

    pipe.radix_tree(n_blocks, 0);

    pipe.edge_count(n_blocks, 0);

    pipe.prefix_sum(n_blocks, 0);

    pipe.octree(n_blocks, 0);
}