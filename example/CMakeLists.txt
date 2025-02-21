# Example cmake project
cmake_minimum_required(VERSION 3.18)
project(tokenizers_cpp_example C CXX)

if(CMAKE_TOOLCHAIN_FILE)
  if(DEFINED include_CMAKE_TOOLCHAIN_FILE)
  else()
    include(${CMAKE_TOOLCHAIN_FILE})
    set(include_CMAKE_TOOLCHAIN_FILE ON)
  endif()
endif()

find_package(sentencepiece CONFIG REQUIRED)

if(DEFINED ENABLE_TORCH)
else()
  set(ENABLE_TORCH ON CACHE BOOL "If enable torch")
endif()

if(ENABLE_TORCH)
  if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
  endif()

  add_definitions(-DENABLE_TORCH)
endif()

include(CheckCXXCompilerFlag)

if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

# include tokenizer cpp as a sub directory
set(TOKENZIER_CPP_PATH ..)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_executable(example example.cc)

target_include_directories(example
  PRIVATE ${TOKENZIER_CPP_PATH}/include
  PUBLIC ${TORCH_INCLUDE_DIRS}
)

# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary
target_link_libraries(example PRIVATE tokenizers_cpp ${TORCH_LIBRARIES})
