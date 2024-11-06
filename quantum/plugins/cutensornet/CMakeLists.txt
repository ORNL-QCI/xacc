set(LIBRARY_NAME xacc-cutensornet)

file(GLOB SRC *.cpp)

usfunctiongetresourcesource(TARGET ${LIBRARY_NAME} OUT SRC)
usfunctiongeneratebundleinit(TARGET ${LIBRARY_NAME} OUT SRC)

set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME}
                             PUBLIC . 
                             ${CUQUANTUM_ROOT}/include
                             ${CUTENSOR_ROOT}/include)

find_library(CUTENSORNET_LIB cutensornet HINTS ${CUQUANTUM_ROOT}/lib)
# First try to locate libcutensor.so.2
# Find the primary cutensor library
find_library(CUTENSOR_LIB cutensor HINTS ${CUTENSOR_ROOT}/lib/11)

# Try to locate the versioned libcutensor.so.2, if available
find_library(CUTENSOR_LIB2 libcutensor.so.2 HINTS ${CUTENSOR_ROOT}/lib/11)

if(NOT CUTENSOR_LIB)
    message(FATAL_ERROR "Could not find libcutensor")
endif()

if(NOT CUTENSOR_LIB2)
    message(FATAL_ERROR "Could not find libcutensor2")
endif()

target_link_libraries(${LIBRARY_NAME}
                        PUBLIC xacc
                               xacc-quantum-gate
          PRIVATE
          ${CUTENSORNET_LIB}
          ${CUTENSOR_LIB}
          ${CUTENSOR_LIB2}
          cudart)

set(_bundle_name xacc_cutensornet)
set_target_properties(${LIBRARY_NAME}
                      PROPERTIES COMPILE_DEFINITIONS
                                 US_BUNDLE_NAME=${_bundle_name}
                                 US_BUNDLE_NAME
                                 ${_bundle_name})

usfunctionembedresources(TARGET
                         ${LIBRARY_NAME}
                         WORKING_DIRECTORY
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         FILES
                         manifest.json)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_compile_definitions(${LIBRARY_NAME} PUBLIC WITH_OPENMP_)
  target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
                         
if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
set_target_properties(${LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "${CUTENSOR_ROOT}/lib/11")              
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

if(XACC_BUILD_TESTS)
  add_subdirectory(tests)
endif()

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)