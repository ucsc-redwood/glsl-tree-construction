#!/bin/bash

# Executable name
executable="bench-gpu"

# Path to local executable
local_executable_path="./build/android/arm64-v8a/release/$executable"

# Path to the executable on the Android device
executable_path="/data/local/tmp/compiled_shaders"

# Push the executable to the Android device
adb push $local_executable_path $executable_path

# Run the executable on the Android device
adb shell "cd $executable_path && ./$executable --benchmark_out=results.txt"

# Print the output of the executable
#adb shell "cat $executable_path/bench_output.txt"
# Remove the executable from the Android device
#adb shell rm $executable_path./