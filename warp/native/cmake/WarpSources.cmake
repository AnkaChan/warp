# WarpSources.cmake
#
# Include this file from your CMakeLists.txt to compile Warp's native C++/CUDA
# sources as part of your own project.
#
# Usage:
#   set(WARP_DIR "path/to/warp")
#   include(${WARP_DIR}/warp/native/cmake/WarpSources.cmake)
#
#   add_executable(my_app ${MY_SRCS} ${WARP_CXX_SRCS} ${WARP_CUDA_SRCS})
#   target_include_directories(my_app PRIVATE ${WARP_INCLUDE_DIR})
#   target_compile_definitions(my_app PRIVATE ${WARP_COMPILE_DEFINITIONS})
#   if(MSVC)
#       target_compile_options(my_app PRIVATE ${WARP_MSVC_OPTIONS})
#   endif()
#   # For CUDA targets:
#   target_compile_options(my_app PRIVATE ${WARP_CUDA_OPTIONS})
#
# Provided variables:
#   WARP_INCLUDE_DIR          - Include directory (warp/native/)
#   WARP_CXX_SRCS             - C++ source files (.cpp)
#   WARP_CUDA_SRCS            - CUDA source files (.cu), empty if CUDA disabled
#   WARP_HEADERS              - All header files
#   WARP_COMPILE_DEFINITIONS  - Required compile definitions
#   WARP_MSVC_OPTIONS         - MSVC warning suppressions
#   WARP_CUDA_OPTIONS         - Recommended NVCC compile flags
#   WARP_CUDA_FOUND           - TRUE if CUDA was found and enabled

# Resolve the native source directory relative to this cmake file
get_filename_component(_WARP_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(WARP_INCLUDE_DIR "${_WARP_NATIVE_DIR}")

# --- Options ---
option(WARP_ENABLE_CUDA "Enable CUDA support for Warp sources" ON)
option(WARP_ENABLE_MATHDX "Enable MathDx support for Warp sources" OFF)

# --- Detect CUDA ---
set(WARP_CUDA_FOUND FALSE)
if(WARP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(WARP_CUDA_FOUND TRUE)
        endif()
    endif()
endif()

# --- C++ source files ---
# NOTE: bvh.cpp and scan.cpp are #include'd by warp.cpp, so they must NOT be
# compiled separately to avoid duplicate symbol errors.
set(WARP_CXX_SRCS
    "${_WARP_NATIVE_DIR}/crt.cpp"
    "${_WARP_NATIVE_DIR}/error.cpp"
    "${_WARP_NATIVE_DIR}/warp.cpp"
    "${_WARP_NATIVE_DIR}/mesh.cpp"
    "${_WARP_NATIVE_DIR}/hashgrid.cpp"
    "${_WARP_NATIVE_DIR}/volume.cpp"
    "${_WARP_NATIVE_DIR}/sort.cpp"
    "${_WARP_NATIVE_DIR}/reduce.cpp"
    "${_WARP_NATIVE_DIR}/runlength_encode.cpp"
    "${_WARP_NATIVE_DIR}/sparse.cpp"
    "${_WARP_NATIVE_DIR}/coloring.cpp"
    "${_WARP_NATIVE_DIR}/cuda_util.cpp"
    "${_WARP_NATIVE_DIR}/mathdx.cpp"
    "${_WARP_NATIVE_DIR}/texture.cpp"
)

# --- CUDA source files ---
set(WARP_CUDA_SRCS "")
if(WARP_CUDA_FOUND)
    set(WARP_CUDA_SRCS
        "${_WARP_NATIVE_DIR}/bvh.cu"
        "${_WARP_NATIVE_DIR}/mesh.cu"
        "${_WARP_NATIVE_DIR}/sort.cu"
        "${_WARP_NATIVE_DIR}/hashgrid.cu"
        "${_WARP_NATIVE_DIR}/reduce.cu"
        "${_WARP_NATIVE_DIR}/runlength_encode.cu"
        "${_WARP_NATIVE_DIR}/scan.cu"
        "${_WARP_NATIVE_DIR}/sparse.cu"
        "${_WARP_NATIVE_DIR}/volume.cu"
        "${_WARP_NATIVE_DIR}/volume_builder.cu"
        "${_WARP_NATIVE_DIR}/warp.cu"
    )
endif()

# --- Header files ---
file(GLOB WARP_HEADERS "${_WARP_NATIVE_DIR}/*.h")

# --- Compile definitions ---
set(WARP_COMPILE_DEFINITIONS
    WP_ENABLE_CUDA=$<BOOL:${WARP_CUDA_FOUND}>
    WP_ENABLE_CUDA_COMPATIBILITY=0
    WP_ENABLE_DEBUG=$<IF:$<CONFIG:Debug>,1,0>
    WP_ENABLE_MATHDX=$<BOOL:${WARP_ENABLE_MATHDX}>
)

# --- MSVC options ---
set(WARP_MSVC_OPTIONS /wd4804 /wd5999)

# --- CUDA compile options ---
set(WARP_CUDA_OPTIONS
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
)
if(MSVC AND WARP_CUDA_FOUND)
    list(APPEND WARP_CUDA_OPTIONS
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/wd4804">
        $<$<COMPILE_LANGUAGE:CXX>:/wd4804>
        $<$<COMPILE_LANGUAGE:CXX>:/wd5999>
    )
endif()

# --- CUDA link libraries ---
# warp.cu uses nvrtc and nvPTXCompiler
set(WARP_CUDA_LINK_LIBS "")
if(WARP_CUDA_FOUND)
    set(WARP_CUDA_LINK_LIBS
        CUDA::cudart_static
        CUDA::cuda_driver
        CUDA::nvrtc_static
        CUDA::nvrtc_builtins_static
        CUDA::nvptxcompiler_static
    )
    # nvJitLink is available in CUDA 12+
    if(TARGET CUDA::nvJitLink_static)
        list(APPEND WARP_CUDA_LINK_LIBS CUDA::nvJitLink_static)
    endif()
endif()

# --- Status message ---
message(STATUS "Warp native sources: ${_WARP_NATIVE_DIR}")
message(STATUS "  CUDA: ${WARP_CUDA_FOUND}")
message(STATUS "  CXX sources: ${WARP_CXX_SRCS}")
if(WARP_CUDA_FOUND)
    message(STATUS "  CUDA sources: ${WARP_CUDA_SRCS}")
endif()
