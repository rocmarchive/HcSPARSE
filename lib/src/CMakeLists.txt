
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


#Setting a Project name
SET(PROJECT_NAME "hcsparse")
SET(PROJECT_NAME_EXT "hipsparse")
SET(ROCM_PATH "/opt/rocm/")

PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME : ${PROJECT_NAME}")

SET(HIP_SUPPORT $ENV{HIP_SUPPORT})

# Find HIP
set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
       set (HIP_PATH ${ROCM_PATH}/hip)
endif()

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)
MESSAGE (STATUS "HIP_PATH : ${HIP_PATH}")

# Build hcsparse and hipsparse from source on AMD platform
IF (${PLATFORM} MATCHES "hcc")

# Find HCC compiler
  FIND_PACKAGE(HC++ 1.0 REQUIRED)
#Setting a variable for source files
  SET (INIT_SRC
     hcsparse_init.cpp
     )

#Setting a variable for source files
  SET (SRCS ${INIT_SRC})

  # Choice to take compilation flags from source or package
  if(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --build --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  else(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  endif(EXISTS ${MCWHCCBUILD})
 
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include")
  set (HCC_LDFLAGS "${HCC_LDFLAGS}")

  IF (${HIP_SUPPORT} MATCHES "on") 
    set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${ROCM_PATH}/include -I${HIP_PATH}/include")
    set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc")
  ENDIF()

  FOREACH(src_file ${SRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()

  #Generating hcfft shared object
  ADD_LIBRARY("${PROJECT_NAME}" SHARED  ${SRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}" hc_am)

  INSTALL(TARGETS "${PROJECT_NAME}" 
   RUNTIME DESTINATION lib
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
   PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )

  INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcsparse.h" "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcsparse_struct.h" DESTINATION include)

  IF (${HIP_SUPPORT} MATCHES "on")
    SET (HIPSRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipsparse.cpp)
 
    # Setting header dependencies for hipblas library generation
    FOREACH(src_file ${HIPSRCS})
          SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
    ENDFOREACH()
    
    #Generating hipblas shared object
    ADD_LIBRARY("${PROJECT_NAME_EXT}_hcc" SHARED ${HIPSRCS})
    SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}_hcc" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
    TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}_hcc" hc_am)

    INSTALL(TARGETS "${PROJECT_NAME_EXT}_hcc" 
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
    )
    INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include) 
  ENDIF()
ELSEIF (${PLATFORM} MATCHES "nvcc")
  MESSAGE(STATUS "Building hipsparse.cpp")
  SET(HIPSRCS ${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipsparse.cpp)

  set (CXXFLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -I/usr/local/cuda/include")
  set (LDFLAGS "-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib")
  FOREACH(src_file ${HIPSRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CXXFLAGS} ")
  ENDFOREACH()
  ADD_LIBRARY("${PROJECT_NAME_EXT}_nvcc" SHARED ${HIPFFTSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}_nvcc" APPEND_STRING PROPERTY LINK_FLAGS " ${LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}_nvcc" cudart cufft)
  
  INSTALL(TARGETS "${PROJECT_NAME_EXT}_nvcc" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )

  INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include)
ENDIF()


