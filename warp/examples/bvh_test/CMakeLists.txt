cmake_minimum_required(VERSION 3.28)
project(bvh_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Include warp sources
set(WARP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include(${WARP_DIR}/warp/native/cmake/WarpSources.cmake)

# Build warp CXX sources as an object library
add_library(warp_cxx OBJECT ${WARP_CXX_SRCS})
target_include_directories(warp_cxx PUBLIC ${WARP_INCLUDE_DIR})
target_compile_definitions(warp_cxx PUBLIC ${WARP_COMPILE_DEFINITIONS})
set_target_properties(warp_cxx PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(warp_cxx PRIVATE ${WARP_MSVC_OPTIONS})
endif()
if(WARP_CUDA_FOUND)
    target_include_directories(warp_cxx PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Build warp CUDA sources as a static library (avoids VS output path issues)
if(WARP_CUDA_FOUND)
    add_library(warp_cuda STATIC ${WARP_CUDA_SRCS})
    target_include_directories(warp_cuda PUBLIC ${WARP_INCLUDE_DIR})
    target_compile_definitions(warp_cuda PUBLIC ${WARP_COMPILE_DEFINITIONS})
    target_compile_options(warp_cuda PRIVATE ${WARP_CUDA_OPTIONS})
    set_target_properties(warp_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(warp_cuda PRIVATE ${WARP_CUDA_LINK_LIBS})
endif()

# Create executable
add_executable(bvh_test main.cpp $<TARGET_OBJECTS:warp_cxx>)
target_include_directories(bvh_test PRIVATE ${WARP_INCLUDE_DIR})
target_compile_definitions(bvh_test PRIVATE ${WARP_COMPILE_DEFINITIONS})
if(MSVC)
    target_compile_options(bvh_test PRIVATE ${WARP_MSVC_OPTIONS})
endif()

if(WARP_CUDA_FOUND)
    target_include_directories(bvh_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(bvh_test PRIVATE warp_cuda ${WARP_CUDA_LINK_LIBS})
    set_target_properties(bvh_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
