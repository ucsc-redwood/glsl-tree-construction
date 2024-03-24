#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <vulkan/vulkan.h>
#include "app_params.hpp"
#include "naive_pipe.hpp"

namespace bm = benchmark;

#define BUFFER_ELEMENTS 131072
#define MAX_BLOCKS 128

class PipeBenchmark : public Pipe
{

public:
    PipeBenchmark(AppParams app_params) : Pipe(app_params) {}

    void BM_GPU_Morton(bm::State &st)
    {
        int n_blocks = 128;
        std::fill_n(u_points, params_.n, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill_n(u_morton_keys, params_.n, 0);

        for (auto _ : st)
        {
            morton(n_blocks, 0);
            st.SetIterationTime(time());
        }
    }

    void BM_GPU_Sort(bm::State &st)
    {
        int n_blocks = 128;
        constexpr auto radix = 256;
        constexpr auto passes = 4;
        const auto binning_thread_blocks = (params_.n + 7680 - 1) / 7680;

        uint32_t *copy_of_u_morton_keys = new uint32_t[params_.n];
        std::copy(u_morton_keys, u_morton_keys + params_.n, copy_of_u_morton_keys);

        std::fill_n(sort_tmp.u_sort_alt, params_.n, 0);
        std::fill_n(sort_tmp.u_global_histogram, radix * passes, 0);
        std::fill_n(sort_tmp.u_pass_histogram, radix * binning_thread_blocks, glm::uvec4(0, 0, 0, 0));
        std::fill_n(sort_tmp.u_index, 4, 0);

        for (auto _ : st)
        {
            radix_sort(n_blocks, 0);
            st.SetIterationTime(time());
        }
        // if (n_blocks != MAX_BLOCKS)
        // {
        //     u_morton_keys = copy_of_u_morton_keys;
        // }
    }

    void BM_GPU_Unique(bm::State &st)
    {
        int n_blocks = st.range(0);

        uint32_t *copy_of_u_morton_keys = new uint32_t[params_.n];
        std::copy(u_morton_keys, u_morton_keys + params_.n, copy_of_u_morton_keys);

        std::fill_n(u_unique_morton_keys, params_.n, 0);
        std::fill_n(unique_tmp.contributions, params_.n, 0);
        std::fill_n(unique_tmp.reductions, n_blocks, 0);
        std::fill_n(unique_tmp.index, 1, 0);

        for (auto _ : st)
        {
            unique(n_blocks, 0);
            st.SetIterationTime(time());
        }
        if (n_blocks != MAX_BLOCKS)
        {
            u_morton_keys = copy_of_u_morton_keys;
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
        ->Range(1, 1)
        ->ArgName("GridSize");

    benchmark::RegisterBenchmark("BM_GPU_Sort", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Sort(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, 1)
        ->ArgName("GridSize");

    benchmark::RegisterBenchmark("BM_GPU_Unique", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Unique(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, MAX_BLOCKS)
        ->ArgName("GridSize");
}

int main(int argc, char **argv)
{
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
}
