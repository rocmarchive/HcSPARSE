# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

#Setting a Project name
SET(PROJECT_NAME "libhcsparsetest")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

#Setting a variable for source files
SET (TESTSRCS 
     scale_float_test.cpp
     scale_double_test.cpp
     axpby_float_test.cpp
     axpby_double_test.cpp
    )

SET (PREFIX "$ENV{MCWCPPAMPBUILD}")
SET (CLANG_AMP "${PREFIX}/compiler/bin/clang++")
SET (CLAMP_CONFIG "${PREFIX}/build/Release/bin/clamp-config")

execute_process(COMMAND ${CLAMP_CONFIG} --build --cxxflags
                            OUTPUT_VARIABLE CLAMP_CXXFLAGS)

string(STRIP "${CLAMP_CXXFLAGS}" CLAMP_CXXFLAGS)
set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS}")

execute_process(COMMAND ${CLAMP_CONFIG} --build --ldflags
                            OUTPUT_VARIABLE CLAMP_LDFLAGS)

string(STRIP "${CLAMP_LDFLAGS}" CLAMP_LDFLAGS)
set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -Wall -Wno-deprecated-register -Wdeprecated-declarations")
set (CLAMP_LDFLAGS "${CLAMP_LDFLAGS}")

set (LINK "-lhcsparse")

FOREACH(test_file ${TESTSRCS})
  SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
  STRING (REPLACE "_test.cpp" "" testname ${test_file}) 
  ADD_EXECUTABLE(Test_Executables/${testname} ${test_file})
  SET_PROPERTY(TARGET Test_Executables/${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${LINK}")
ENDFOREACH()
