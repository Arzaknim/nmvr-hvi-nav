cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(simulation2d)

add_compile_options(-std=c++11)

option(WITH_AVX "use avx" OFF)

if (WITH_AVX)
  add_compile_options(-mavx)
  MESSAGE(STATUS "WITH_AVX")
  add_definitions(-DUSE_AVX)
endif()

find_package(Eigen3 REQUIRED)

# Modern Python find method (replaces old PythonInterp/PythonLibs)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Modern Boost line
find_package(Boost REQUIRED COMPONENTS system python3)


#find_package(SDL2 REQUIRED)

#include_directories(${SDL2_INCLUDE_DIRS})
#set(SDL2_LIBRARIES "-L${SDL2_LIBDIR} -lSDL2")


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST )
include_directories(${PROJECT_SOURCE_DIR}/include )
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

#add_executable(${PROJECT_NAME} ${SRC_LIST})
#target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

#PYTHON_ADD_MODULE(pysim2d SHARED bindings/pysim2d.cpp ${SRC_LIST})
add_library(pysim2d MODULE bindings/pysim2d.cpp ${SRC_LIST})

target_link_libraries(pysim2d
    PRIVATE
        Boost::python3        # or Boost::python3 / Boost::python311 depending on your setup
        Python3::Python
)

set_target_properties(pysim2d PROPERTIES
    PREFIX ""                                   # remove "lib" prefix so module is importable
    SUFFIX ".so"  # proper .so / .pyd suffix
)