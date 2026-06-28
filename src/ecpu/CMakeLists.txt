cmake_minimum_required(VERSION 3.16)
project(piLM_ecpu C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

set(ECPU_SOURCES
    src/core/ecpu.c
    src/device/edevice_registry.c
    src/device/ecpu_device.c
    src/device/cuda_device.c
    src/dispatch/isa_detect.c
    src/kernel/kernels.c
    src/eram/eram.c
)

add_library(ecpu SHARED ${ECPU_SOURCES})
target_include_directories(ecpu PUBLIC include)

if(MSVC)
    set_target_properties(ecpu PROPERTIES OUTPUT_NAME "ecpu" SUFFIX ".dll")
else()
    set_target_properties(ecpu PROPERTIES OUTPUT_NAME "ecpu")
endif()