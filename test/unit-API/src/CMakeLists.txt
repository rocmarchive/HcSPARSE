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
SET (TESTSRCS 
    csrmm_float_test_API.cpp
    csrmm_double_test_API.cpp
    csr_dense_conv_float_test_API.cpp
    csr_dense_conv_double_test_API.cpp
    csr_coo_conv_float_test_API.cpp
    csc_dense_conv_float_test_API.cpp
    csc_dense_conv_double_test_API.cpp
    nnz_float_test_API.cpp
    nnz_double_test_API.cpp
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

  SET (LINK "-lhcsparse")

  FOREACH(test_file ${TESTSRCS})
    SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS}")
    STRING( REPLACE ".cpp" "" testname ${test_file} )
    ADD_EXECUTABLE( ../bin/${testname} ${test_file} )
    TARGET_LINK_LIBRARIES("../bin/${testname}" hc_am)
    SET_PROPERTY(TARGET ../bin/${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
  ENDFOREACH()

