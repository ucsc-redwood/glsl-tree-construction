package("benchmark")
    set_kind("library")
    add_deps("cmake") 
    set_urls("https://android.googlesource.com/platform/external/google-benchmark")
    add_versions("v1.5.0", "06b4a070156a9333549468e67923a3a16c8f541b") 

    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=on")
        table.insert(configs, "-DHAVE_THREAD_SAFETY_ATTRIBUTES=0")
        import("package.tools.cmake").install(package, configs)
    end)
package_end()

add_requires("benchmark")

target("bench-gpu")
    set_kind("binary")
    set_plat("android")
    --set_arch("arm64-v8a") -- Wont let us build google-benchmark if it is set
    add_includedirs("../include")
    add_headerfiles("../include/*.hpp", "../include/**/*.hpp", "../include/*.h")
    add_files("benchmark.cpp", "../src/**/*.cpp", "../include/*.c")
    add_packages("benchmark", "vulkan-headers", "glm", "vulkan-validationlayers")