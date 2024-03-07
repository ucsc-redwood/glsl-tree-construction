{
    files = {
        "build/.objs/app/linux/arm64-v8a/debug/src/main.cpp.o"
    },
    values = {
        "/usr/bin/g++",
        {
            "-L/home/zheyuan/.xmake/packages/s/spirv-cross/1.3.268+0/d7a93afe605f48d1a4ffa8a3b70669c3/lib",
            "-lspirv-cross-c",
            "-lspirv-cross-cpp",
            "-lspirv-cross-reflect",
            "-lspirv-cross-msl",
            "-lspirv-cross-util",
            "-lspirv-cross-hlsl",
            "-lspirv-cross-glsl",
            "-lspirv-cross-core",
            "-lvulkan",
            "-lpthread"
        }
    }
}