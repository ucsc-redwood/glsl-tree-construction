#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <vulkan/vulkan.h>
#include "app_params.hpp"
#include "init.hpp"
#include "morton.hpp"
#include "radixsort.hpp"
#include "radix_tree.hpp"
#include "unique.hpp"
#include "edge_count.hpp"
#include "prefix_sum.hpp"
#include "application.hpp"

namespace bm = benchmark;

#define BUFFER_ELEMENTS 131072

static AppParams app_params;

class PipelineBenchmark : public ApplicationBase
{
public:
    void BM_GPU_Morton(bm::State &st)
    {
        int num_blocks = st.range(0);

        void *mapped;
        glm::vec4 *u_points;
        uint32_t *u_morton_keys;
        VkBuffer u_points_buffer;
        VkBuffer u_morton_keys_buffer;
        VkDeviceMemory u_points_memory;
        VkDeviceMemory u_morton_keys_memory;

        create_shared_empty_storage_buffer(app_params.n * sizeof(glm::vec4), &u_points_buffer, &u_points_memory, &mapped);
        u_points = static_cast<glm::vec4 *>(mapped);
        std::fill_n(u_points, app_params.n, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));

        create_shared_empty_storage_buffer(app_params.n * sizeof(uint32_t), &u_morton_keys_buffer, &u_morton_keys_memory, &mapped);
        u_morton_keys = static_cast<uint32_t *>(mapped);
        std::fill_n(u_morton_keys, app_params.n, 0);

        Morton morton_stage = Morton();

        for (auto _ : st)
        {
            morton_stage.run(num_blocks, 0, u_points, u_morton_keys, u_points_buffer, u_morton_keys_buffer, app_params.n, app_params.min_coord, app_params.getRange());
            st.SetIterationTime(morton_stage.time());
        }

        vkUnmapMemory(singleton.device, u_points_memory);
        vkDestroyBuffer(singleton.device, u_points_buffer, nullptr);
        vkFreeMemory(singleton.device, u_points_memory, nullptr);

        vkUnmapMemory(singleton.device, u_morton_keys_memory);
        vkDestroyBuffer(singleton.device, u_morton_keys_buffer, nullptr);
        vkFreeMemory(singleton.device, u_morton_keys_memory, nullptr);
    }
};

PipelineBenchmark BenchmarkInstance;

static void RegisterBenchmarks()
{
    benchmark::RegisterBenchmark("BM_GPU_Morton", [](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Morton(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, 128)
        ->ArgName("GridSize");
}

int main(int argc, char **argv)
{
    RegisterBenchmarks();

    // Define the parameters
    app_params.n = BUFFER_ELEMENTS;
    app_params.min_coord = 0.0f;
    app_params.max_coord = 1.0f;
    app_params.seed = 114514;
    app_params.n_threads = 4;

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    bm::RunSpecifiedBenchmarks();
}
