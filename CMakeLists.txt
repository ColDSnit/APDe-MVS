cmake_minimum_required(VERSION 3.23.0)

project(APD LANGUAGES CXX CUDA)

# -------------------------------
# Global language/build settings
# -------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Blackwell RTX 50xx default for this stack unless overridden externally.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# -------------------------------
# Windows MSVC/NVCC flag hygiene
# -------------------------------
if(MSVC)
  # Replace /Zi -> /Z7 to avoid NVCC + /Fd,/FS tokenisation issues.
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

  set(CMAKE_CUDA_COMPILE_PDB_NAME "")
  set(CMAKE_CUDA_COMPILE_PDB_OUTPUT_DIRECTORY "")

  if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")
  endif()
endif()

# -------------------------------
# Dependencies
# -------------------------------
find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)

find_package(CUDAToolkit REQUIRED)

# -------------------------------
# Target
# -------------------------------
add_executable(APD
  main.h
  APD.h
  APD.cpp
  APD.cu
  main.cpp
  ThreadPool.h
)

target_include_directories(APD PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(APD PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>
  $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

if(MSVC)
  target_compile_options(APD PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
  )
else()
  target_compile_options(APD PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

target_link_libraries(APD PRIVATE
  ${OpenCV_LIBS}
  Boost::filesystem
  Boost::system
  Boost::program_options
  CUDA::cudart
  CUDA::curand
)

message(STATUS "APDe-MVS Windows patch CMake active")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")