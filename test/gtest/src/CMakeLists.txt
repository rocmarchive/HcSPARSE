#Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

#Setting a Project name
SET(PROJECT_NAME "libhcsparsetest")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
FILE(GLOB TESTSRCS
          gtest-all.cpp
          gtest_main.cpp 
          axpy_float_test.cpp
          axpy_double_test.cpp
          axpby_float_test.cpp
          axpby_double_test.cpp
          dot_float_test.cpp
          dot_double_test.cpp
          scale_float_test.cpp
          scale_double_test.cpp
          reduce_float_test.cpp
          reduce_double_test.cpp
          reduce_int_test.cpp
          transform_float_test.cpp
          transform_double_test.cpp
          nrm2_float_test.cpp
          nrm1_float_test.cpp
          csrmv_float_test.cpp
          csrmv_double_test.cpp
          csrmv_adaptive_float_test.cpp
          csrmv_adaptive_double_test.cpp
          bicgStab_noprecond_float_test.cpp
          bicgStab_noprecond_double_test.cpp
          csr2coo_float_test.cpp
          coo2csr_float_test.cpp
          csr_dense_conv_float_test.cpp
          csr_dense_conv_double_test.cpp
          csrmm_float_test.cpp
          csrmm_double_test.cpp
          spcsrmm_float_test.cpp
          )


# MCW HCC Specific. Version >= 0.3 is Must
ADD_DEFINITIONS(-DLINUX)
  if(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                             OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --build --ldflags 
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  else(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                             OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --install --ldflags
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  endif(EXISTS ${MCWHCCBUILD})

  SET(HCSPARSE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
  SET(HCSPARSE_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src/")
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HCSPARSE_INCLUDE_PATH}")
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCSPARSE_LIBRARY_PATH}")

if(POLICY CMP0037)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0037 OLD)
endif()

SET (LINK "-lhcsparse")
SET ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin/")
SET_PROPERTY(SOURCE ${TESTSRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} -DGTEST_HAS_TR1_TUPLE=0")
ADD_EXECUTABLE( unittest ${TESTSRCS})
SET_PROPERTY(TARGET unittest APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
TARGET_LINK_LIBRARIES(unittest hcsparse hc_am)
add_test(NAME unit COMMAND unittest)

if(POLICY CMP0037)
  cmake_policy(POP)
endif()
