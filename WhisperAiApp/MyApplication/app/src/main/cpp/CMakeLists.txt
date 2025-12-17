cmake_minimum_required(VERSION 3.22.1)
project("myapplication")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(whispercpp)

add_library(native-lib SHARED native-lib.cpp)

find_library(log-lib log)
find_library(android-lib android)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(native-lib whisper ${log-lib} ${android-lib} Threads::Threads)
