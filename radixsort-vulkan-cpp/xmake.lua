set_project("Tree Construction")

-- Vulkan related
add_requires("spirv-cross", "glm")
add_requires("vulkansdk", {system = true})

-- Others
add_requires("spdlog", "cli11")

add_rules("mode.debug", "mode.release")

set_languages("c++20")
set_warnings("allextra")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end

if is_plat("windows") then
elseif is_plat("linux") then
    before_build(function(target)
        os.exec("python3 compile_shaders.py")
        -- os.exec("./compile_shaders.sh")
    end)
end


after_build(function(target)
    platform = os.host()
    arch = os.arch()
    build_path = ""
    if is_mode("release") then
        build_path = "$(buildir)/" .. platform .. "/" .. arch .. "/release/"
    else
        build_path = "$(buildir)/" .. platform .. "/" .. arch .. "/debug/"
    end
    os.cp("shaders/compiled_shaders/**.spv", build_path)
    print("Copied compiled shaders to " .. build_path)
end)
includes("benchmarks")
target("app")
set_default(true)
set_kind("binary")
add_includedirs("include")
add_headerfiles("include/*.hpp", "include/**/*.hpp")
add_files("src/main.cpp", "src/**/*.cpp")
add_packages( "spirv-cross", "glm",
             "vulkansdk", "spdlog", "cli11")
