#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "volk.h"
#include "app_params.hpp"
#include "init.hpp"
#include "morton.hpp"
#include "radixsort.hpp"
#include "radix_tree.hpp"
#include "unique.hpp"
#include "edge_count.hpp"
#include "prefix_sum.hpp"

namespace bm = benchmark;

#define BUFFER_ELEMENTS 131072

static AppParams app_params;

void BM_GPU_Morton(bm::State &st)
{
    app_params.n_blocks = st.range(0);

    std::vector<glm::vec4> data(BUFFER_ELEMENTS, glm::vec4(1, 1, 1, 1));
    std::vector<uint32_t> morton_keys(BUFFER_ELEMENTS, 0);

    Morton morton_stage = Morton();

    for (auto _ : st)
    {
        morton_stage.run(app_params.n_blocks, data.data(), morton_keys.data(),
                         app_params.n, app_params.min_coord, app_params.getRange());
    }
}

BENCHMARK(BM_GPU_Morton)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

void BM_GPU_Sort(bm::State &st)
{
    app_params.n_blocks = st.range(0);
    std::vector<uint32_t> morton_keys(app_params.n, 0);
    std::generate(morton_keys.begin(), morton_keys.end(), [i = app_params.n]() mutable
                  { return --i; });
    RadixSort radixsort_stage = RadixSort();
    for (auto _ : st)
    {
        radixsort_stage.run(app_params.n_blocks, morton_keys.data(), app_params.n);
    }
}

BENCHMARK(BM_GPU_Sort)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

void BM_GPU_Unique(bm::State &st)
{
    app_params.n_blocks = st.range(0);
    std::vector<uint32_t> morton_keys(BUFFER_ELEMENTS, 1);
    std::vector<uint32_t> u_keys(BUFFER_ELEMENTS, 0);
    std::vector<uint32_t> contribution(BUFFER_ELEMENTS, 0);

    Unique unique_stage = Unique();

    for (auto _ : st)
    {
        unique_stage.run(app_params.n_blocks, morton_keys.data(), u_keys.data(),
                         contribution.data(), app_params.n);
    }
}

BENCHMARK(BM_GPU_Unique)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------
void BM_GPU_RadixTree(bm::State &st)
{
    app_params.n_blocks = st.range(0);

    std::vector<uint32_t> u_morton(app_params.n, 0);
    std::vector<uint8_t> prefix_n(app_params.n, 0);
    bool *has_leaf_left = new bool[app_params.n];
    bool *has_leaf_right = new bool[app_params.n];
    std::vector<int> left_child(app_params.n, 0);
    std::vector<int> parents(app_params.n, 0);
    int unique = true;

    std::iota(u_morton.begin(), u_morton.end(), 0);
    std::fill_n(has_leaf_left, app_params.n, false);
    std::fill_n(has_leaf_right, app_params.n, false);

    RadixTree build_radix_tree_stage = RadixTree();

    for (auto _ : st)
    {
        build_radix_tree_stage.run(app_params.n_blocks, u_morton.data(),
                                   prefix_n.data(), has_leaf_left, has_leaf_right, left_child.data(),
                                   parents.data(), unique);
    }
}

BENCHMARK(BM_GPU_RadixTree)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

// void BM_GPU_EdgeCount(bm::State &st)
// {
//     app_params.n_blocks = st.range(0);

//     // TODO: Generate radix tree

//     // const int logical_blocks, uint8_t *prefix_n, int *parent, uint32_t *edge_count, int n_brt_nodes
//     EdgeCount edge_count_stage = EdgeCount();
//     for (auto _ : st)
//     {
//         edge_count_stage.run(app_params.n_blocks, prefix_n.data(),
//                              parents.data(), edge_count.data(), n_brt_nodes);
//     }
// }

// BENCHMARK(BM_GPU_EdgeCount)
//     ->UseManualTime()
//     ->Unit(bm::kMillisecond)
//     ->RangeMultiplier(2)
//     ->Range(1, 128)
//     ->Iterations(300) // takes too long
//     ->ArgName("GridSize");

// ----------------------------------------------------------------------------

void BM_GPU_PrefixSum(bm::State &st)
{
    app_params.n_blocks = st.range(0);

    std::vector<uint32_t> edge_count(app_params.n, 1);

    PrefixSum prefix_sum_stage = PrefixSum();
    for (auto _ : st)
    {
        prefix_sum_stage.run(app_params.n_blocks, edge_count.data(), app_params.n);
    }
}

BENCHMARK(BM_GPU_PrefixSum)->UseManualTime()->Unit(bm::kMillisecond);

// ----------------------------------------------------------------------------

// void BM_GPU_Octree(bm::State &st)
// {
//     app_params.n_blocks = st.range(0);
//     for (auto _ : st)
//     {
//     }

// }

// BENCHMARK(BM_GPU_Octree)
//     ->Unit(bm::kMillisecond)
//     ->UseManualTime()
//     ->RangeMultiplier(2)
//     ->Range(1, 128)
//     ->ArgName("GridSize");

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Initialize volk
    if (volkInitialize() != VK_SUCCESS)
    {
        std::cerr << "Failed to initialize volk!" << std::endl;
        return EXIT_FAILURE;
    }

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