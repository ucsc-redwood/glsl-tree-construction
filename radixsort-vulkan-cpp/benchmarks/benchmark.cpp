#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <vulkan/vulkan.h>
#include "app_params.hpp"
#include "naive_pipe.hpp"

namespace bm = benchmark;

#define BUFFER_ELEMENTS 131072

class PipelineBenchmark // :public ApplicationBase
{
private:
    Pipe pipe;

public:
    PipelineBenchmark(AppParams app_params) : pipe(app_params)
    {
        pipe.allocate();
        pipe.init(app_params.n_blocks, 0);
    }

    void BM_GPU_Morton(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            pipe.morton(n_blocks, 0);
            st.SetIterationTime(pipe.time());
        }
    }

    void BM_GPU_Sort(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            pipe.radix_sort(n_blocks, 0);
            st.SetIterationTime(pipe.time());
        }
    }

    void BM_GPU_Unique(bm::State &st)
    {
        int n_blocks = st.range(0);
        for (auto _ : st)
        {
            pipe.unique(n_blocks, 0);
            st.SetIterationTime(pipe.time());
        }
    }
};

static void RegisterBenchmarks(PipelineBenchmark &BenchmarkInstance)
{
    benchmark::RegisterBenchmark("BM_GPU_Morton", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Morton(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, 128)
        ->ArgName("GridSize");

    benchmark::RegisterBenchmark("BM_GPU_Sort", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Sort(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, 128)
        ->ArgName("GridSize");

    // Adding the new benchmark registration for BM_GPU_Unique
    benchmark::RegisterBenchmark("BM_GPU_Unique", [&BenchmarkInstance](benchmark::State &st)
                                 { BenchmarkInstance.BM_GPU_Unique(st); })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->RangeMultiplier(2)
        ->Range(1, 128)
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

    PipelineBenchmark Benchmark(app_params);

    RegisterBenchmarks(Benchmark);

    // Run the benchmarks
    bm::Initialize(&argc, argv);
    bm::RunSpecifiedBenchmarks();
}
