{
    depfiles_gcc = "main.o: src/main.cpp include/vma_usage.hpp include/radixsort.hpp  include/application.hpp include/core/VulkanTools.h\
",
    values = {
        "/usr/bin/gcc",
        {
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-O3",
            "-std=c++2a",
            "-Iinclude",
            "-isystem",
            "/home/zheyuan/.xmake/packages/v/vulkan-memory-allocator/v3.0.1/f7da89be499947a189b29a8c3550dab4/include",
            "-isystem",
            "/home/zheyuan/.xmake/packages/s/spirv-cross/1.3.268+0/d7a93afe605f48d1a4ffa8a3b70669c3/include",
            "-isystem",
            "/home/zheyuan/.xmake/packages/s/spdlog/v1.13.0/1e5e36d080174116879d672ca4ac2478/include",
            "-isystem",
            "/home/zheyuan/.xmake/packages/c/cli11/v2.3.2/dbc871d99f0940df8b0bd25e622e015e/include",
            "-DNDEBUG"
        }
    },
    files = {
        "src/main.cpp"
    }
}