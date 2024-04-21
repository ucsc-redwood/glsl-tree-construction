#!/bin/bash

# Executable name
executable="bench-gpu"

# Path to local executable
local_executable_path="./build/android/armeabi-v7a/release/$executable"

# Path to the executable on the Android device
executable_path="/data/local/tmp/compiled_shaders"

# Push the executable to the Android device
adb push $local_executable_path $executable_path

# Check if the -v flag is provided
if [[ $1 == "-v" ]]; then
    # Run the executable on the Android device and print the output
    adb shell "cd $executable_path && ./$executable --benchmark_out=bench_output.txt --benchmark_out_format=console"
else
    # Run the executable on the Android device and redirect the output to /dev/null
    adb shell "cd $executable_path && ./$executable --benchmark_out=bench_output.txt --benchmark_out_format=console > /dev/null 2>&1"
fi

# Print the output of the executable
adb shell "cat $executable_path/bench_output.txt"

# Remove the executable from the Android device
#adb shell rm $executable_path./