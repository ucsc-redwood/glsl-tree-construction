set_project("Tree Construction")

-- Vulkan related
add_requires( "glm")
add_requires("vulkan-headers")

-- Others
-- add_requires("spdlog", "cli11")

add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("all")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end

before_build(function(target)
    os.exec("python3 compile_shaders.py")
end)


target("app")
set_default(true)
set_plat("android")
set_arch("arm64-v8a")
set_kind("binary")
add_includedirs("include")
add_headerfiles("include/*.hpp", "include/**/*.hpp", "include/*.h")
add_files("src/main.cpp", "src/**/*.cpp", "include/*.c")
add_packages("vulkan-headers", "glm", "vulkan-validationlayers")


after_build(function(target)
    platform = os.host()
    arch = os.arch()
    build_path = ""
    local symsdir = path.join("$(buildir)", "$(plat)", "syms", "$(mode)", "$(arch)")
    local validationlayers = target:pkg("vulkan-validationlayers")
    if validationlayers then
        local arch = target:arch()
        os.vcp(path.join(validationlayers:installdir(), "lib", arch, "*.so"), path.join(os.scriptdir(), "..", "libs", arch))
    end
    if is_mode("release") then
        build_path = "$(buildir)/" .. platform .. "/" .. arch .. "/release/"
    else
        build_path = "$(buildir)/" .. platform .. "/" .. arch .. "/debug/"
    end
    os.cp("shaders/compiled_shaders/**.spv", build_path)
    print("Copied compiled shaders to " .. build_path)
    
end)

