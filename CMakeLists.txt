cmake_minimum_required(VERSION 3.16)

option(SINKHORN_BUILD_DEVICE "Build Ascend C kernel and launch adapter" OFF)
if(SINKHORN_BUILD_DEVICE)
    find_package(ASC REQUIRED)
    project(sinkhorn_norm LANGUAGES ASC CXX)
else()
    project(sinkhorn_norm LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(sinkhorn_tiling STATIC src/host/tiling.cpp src/host/tiling_key.cpp)
target_include_directories(sinkhorn_tiling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(sinkhorn_tiling PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(sinkhorn_tiling PRIVATE -Wall -Wextra -Wpedantic)

if(SINKHORN_BUILD_DEVICE)
    set(NPU_ARCH "dav-2201" CACHE STRING "Ascend target architecture")
    add_library(sinkhorn_norm SHARED src/kernel/sinkhorn_norm.asc src/host/launch.cpp)
    set_target_properties(sinkhorn_norm PROPERTIES LINKER_LANGUAGE ASC)
    target_include_directories(sinkhorn_norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    find_path(SINKHORN_CANN_INCLUDE_DIR tiling/platform/platform_ascendc.h
        HINTS "$ENV{ASCEND_HOME_PATH}/include" "$ENV{ASCEND_CANN_PACKAGE_PATH}/include")
    if(NOT SINKHORN_CANN_INCLUDE_DIR)
        message(FATAL_ERROR "CANN host headers not found; source ascend-toolkit/set_env.sh")
    endif()
    target_include_directories(sinkhorn_norm PRIVATE ${SINKHORN_CANN_INCLUDE_DIR})
    target_compile_options(sinkhorn_norm PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)
    target_link_libraries(sinkhorn_norm PRIVATE sinkhorn_tiling
        tiling_api register platform unified_dlog dl m graph_base ascendcl)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(sinkhorn_tiling_test tests/tiling_test.cpp)
    target_link_libraries(sinkhorn_tiling_test PRIVATE sinkhorn_tiling)
    add_test(NAME sinkhorn_tiling COMMAND sinkhorn_tiling_test)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME sinkhorn_reference
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_test.py)
    endif()
endif()
