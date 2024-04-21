#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include "app_params.hpp"
#include "naive_pipe.hpp"

namespace bm = benchmark;

#define BUFFER_ELEMENTS 1920 * 1080
#define MAX_BLOCKS 128
#define ITERATIONS 1

class PipeBenchmark : public Pipe
{

public:
    PipeBenchmark(AppParams app_params) : Pipe(app_params) {}

    void BM_GPU_Morton(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            morton(n_blocks, 0);
            st.SetIterationTime(time());
        }
    }

    //   u_morton_keys, DONE
    //   sort_tmp.u_sort_alt, DONE
    //   sort_tmp.u_global_histogram, DONE
    //   sort_tmp.u_index, DONE
    //   sort_tmp.u_pass_histogram_64, DONE
    //   u_morton_keys_buffer,
    //   sort_tmp.u_sort_alt_buffer,
    //   sort_tmp.u_global_histogram_buffer,
    //   sort_tmp.u_index_buffer,
    //   sort_tmp.u_pass_histogram_64_buffer,

    void BM_GPU_Sort(bm::State &st)
    {
        int n_blocks = st.range(0);
        constexpr auto radix = 256;
        constexpr auto passes = 4;
        const auto max_binning_thread_blocks = ((2 << 25) + 7680 - 1) / 7680;
        int iters = 1;
        int n = params_.n;

        // Allocate memory for the copy
        uint32_t *u_morton_keys_copy = new uint32_t[n];

        // Copy the elements
        std::copy(u_morton_keys, u_morton_keys + n, u_morton_keys_copy);

        for (auto _ : st)
        {
            radix_sort_alt(n_blocks, 0);
            st.SetIterationTime(time());
            // Reset the data
            std::fill_n(sort_tmp.u_sort_alt, params_.n, 0);
            std::fill_n(sort_tmp.u_global_histogram, radix * passes, 0);
            std::fill_n(sort_tmp.u_index, 4, 0);
            std::fill_n(sort_tmp.u_pass_histogram_64, radix * passes * max_binning_thread_blocks, 0);
            std::copy(u_morton_keys_copy, u_morton_keys_copy + n, u_morton_keys);
        }
    }

    void BM_GPU_Unique(bm::State &st)
    {
        int n_blocks = st.range(0);
        uint32_t aligned_size = ((params_.n + 4 - 1) / 4) * 4;
        const uint32_t num_blocks = (aligned_size + PARTITION_SIZE - 1) / PARTITION_SIZE;
        int iters = 1;
        for (auto _ : st)
        {
            unique(n_blocks, 0);
            st.SetIterationTime(time());
            if (n_blocks != MAX_BLOCKS || iters < ITERATIONS)
            {
                std::fill_n(u_unique_morton_keys, params_.n, 0);
                std::fill_n(unique_tmp.contributions, params_.n, 0);
                std::fill_n(unique_tmp.reductions, num_blocks, 0);
                std::fill_n(unique_tmp.index, 1, 0);
            }
            iters++;
        }
    }

    void BM_GPU_RadixTree(bm::State &st)
    {
        int n_blocks = st.range(0);
        // Cache data
        uint32_t *u_copy_morton_keys = (uint32_t *)malloc(params_.n * sizeof(uint32_t));
        memcpy(u_copy_morton_keys, u_unique_morton_keys, params_.n * sizeof(uint32_t));
        int iters = 1;
        for (auto _ : st)
        {
            radix_tree(n_blocks, 0);
            st.SetIterationTime(time());
            // Reset data for next iteration
            if (n_blocks != MAX_BLOCKS || iters < ITERATIONS)
            {
                memcpy(u_unique_morton_keys, u_copy_morton_keys, params_.n * sizeof(uint32_t));
            }
            iters++;
        }
    }

    void BM_GPU_EdgeCount(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            edge_count(n_blocks, 0);
            st.SetIterationTime(time());
        }
    }

    void BM_GPU_PrefixSum(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            prefix_sum(n_blocks, 0);
            st.SetIterationTime(time());
        }
    }

    void BM_GPU_Octree(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            octree(n_blocks, 0);
            st.SetIterationTime(time());
        }
    }
};

static void RegisterBenchmarks(PipeBenchmark &BenchmarkInstance)
{
    benchmark::RegisterBenchmark("BM_GPU_Morton", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Morton(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, MAX_BLOCKS)
        ->Iterations(ITERATIONS)
        ->ArgName("GridSize");

    benchmark::RegisterBenchmark("BM_GPU_Sort", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Sort(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, MAX_BLOCKS)
        ->Iterations(ITERATIONS)
        ->ArgName("GridSize");

    // benchmark::RegisterBenchmark("BM_GPU_Unique", [&BenchmarkInstance](benchmark::State &st)
    //                              { BenchmarkInstance.BM_GPU_Unique(st); })
    //     ->UseManualTime()
    //     ->Unit(benchmark::kMillisecond)
    //     ->RangeMultiplier(2)
    //     ->Range(1, MAX_BLOCKS)
    //     ->Iterations(ITERATIONS)
    //     ->ArgName("GridSize");

    // benchmark::RegisterBenchmark("BM_GPU_RadixTree", [&BenchmarkInstance](benchmark::State &st)
    //                              { BenchmarkInstance.BM_GPU_RadixTree(st); })
    //     ->UseManualTime()
    //     ->Unit(benchmark::kMillisecond)
    //     ->RangeMultiplier(2)
    //     ->Range(1, MAX_BLOCKS)
    //     ->Iterations(ITERATIONS)
    //     ->ArgName("GridSize");

    // benchmark::RegisterBenchmark("BM_GPU_EdgeCount", [&BenchmarkInstance](benchmark::State &st)
    //                              { BenchmarkInstance.BM_GPU_EdgeCount(st); })
    //     ->UseManualTime()
    //     ->Unit(benchmark::kMillisecond)
    //     ->RangeMultiplier(2)
    //     ->Range(1, MAX_BLOCKS)
    //     ->Iterations(ITERATIONS)
    //     ->ArgName("GridSize");

    // benchmark::RegisterBenchmark("BM_GPU_PrefixSum", [&BenchmarkInstance](benchmark::State &st)
    //                              { BenchmarkInstance.BM_GPU_PrefixSum(st); })
    //     ->UseManualTime()
    //     ->Unit(benchmark::kMillisecond)
    //     ->RangeMultiplier(2)
    //     ->Range(1, MAX_BLOCKS)
    //     ->Iterations(ITERATIONS)
    //     ->ArgName("GridSize");

    // benchmark::RegisterBenchmark("BM_GPU_Octree", [&BenchmarkInstance](benchmark::State &st)
    //                              { BenchmarkInstance.BM_GPU_Octree(st); })
    //     ->UseManualTime()
    //     ->Unit(benchmark::kMillisecond)
    //     ->RangeMultiplier(2)
    //     ->Range(1, MAX_BLOCKS)
    //     ->Iterations(ITERATIONS)
    //     ->ArgName("GridSize");
}

int main(int argc, char **argv)
{
    if (volkInitialize() != VK_SUCCESS)
    {
        std::cerr << "Failed to initialize volk!" << std::endl;
        return EXIT_FAILURE;
    }

    AppParams app_params;
    app_params.n = BUFFER_ELEMENTS;
    app_params.min_coord = 0.0f;
    app_params.max_coord = 1.0f;
    app_params.seed = 114514;
    app_params.n_threads = 4;

    PipeBenchmark Benchmark(app_params);

    Benchmark.allocate();
    Benchmark.init(1, 0);

    RegisterBenchmarks(Benchmark);

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    bm::RunSpecifiedBenchmarks();

    // print hello world
    std::cout << "Hello, World!" << std::endl;
}