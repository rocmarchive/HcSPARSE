#!/bin/bash -e
#This script is invoked to install the hcsparse library and test sources

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# CHECK FOR COMPILER PATH
if [ ! -z $MCWHCCBUILD ]
then
  platform="hcc"
  if [ -x "$MCWHCCBUILD/bin/clang++" ]
  then
    cmake_c_compiler="$MCWHCCBUILD/bin/clang"
    cmake_cxx_compiler="$MCWHCCBUILD/bin/clang++"
  fi
elif [ -x "/opt/rocm/hcc/bin/clang++" ]
then
  platform="hcc"
  cmake_c_compiler="/opt/rocm/hcc/bin/clang"
  cmake_cxx_compiler="/opt/rocm/hcc/bin/clang++"
elif [ -x "/usr/local/cuda/bin/nvcc" ];
then
  platform="nvcc"
  cmake_c_compiler="/usr/bin/gcc"
  cmake_cxx_compiler="/usr/bin/g++"
else
  echo "Neither clang  or NVCC compiler found"
  echo "Not an AMD or NVCC compatible stack"
  exit 1
fi

if ( [ ! -z $HIP_PATH ] || [ -x "/opt/rocm/hip/bin/hipcc" ] ); then 
  export HIP_SUPPORT=on
elif ( [ "$platform" = "nvcc" ]); then
  echo "HIP not found. Install latest HIP to continue."
  exit 1
fi

export CLAMP_NOTILECHECK=ON

if [ ! -z $HIP_PATH ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIP_PATH/lib
else
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib
fi

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Help menu
print_help() {
cat <<-HELP
=============================================================================================================================
This script is invoked to install hcblas library and test sources. Please provide the following arguments:

  1) ${green}--test${reset}    Test to enable the library testing. 

=============================================================================================================================
HELP
exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --test=*)
      testing="${1#*=}"
      ;;
    --install)
      install="1"
      ;;
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

if [ "$install" = "1" ]; then
    export INSTALL_OPT=on
fi

set +e
# MAKE BUILD DIR
mkdir $current_work_dir/build
mkdir $current_work_dir/build/test
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

# change to library build
cd $build_dir

if [ "$platform" = "hcc" ]; then
  
   export HCSPARSE_LIBRARY_PATH=$current_work_dir/build/lib/src
   export LD_LIBRARY_PATH=$HCSPARSE_LIBRARY_PATH:$LD_LIBRARY_PATH


# Cmake and make libhcblas: Install hcblas
   cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir
   make package
   make

   if [ "$install" = "1" ]; then
     sudo make install
     cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir/packaging/
   fi

# Test=OFF (Build library and tests)
  if ( [ -z $testing ] ) || ( [ "$testing" = "off" ] ); then
    echo "${green}HCSPARSE Installation Completed!${reset}"
# Test=ON (Build and test the library)
  elif ( [ "$testing" = "on" ] ); then
# Build Tests
     set +e
     mkdir $current_work_dir/build/test/unit_test/bin/
     mkdir $current_work_dir/build/test/gtest/bin/
     set -e
     
     cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
     make

# Invoke hc unit test script
     printf "* UNIT API TESTS *\n"
     printf "******************\n"
     cd $current_work_dir/build/test/gtest/bin/
     ./unittest
    
     if [ $HIP_SUPPORT = "on" ]; then
# Invoke hip unit test script
       printf "* UNIT HIP TESTS *\n"
       printf "******************\n"
       cd $current_work_dir/build/test/unit-hip/src/bin/
       ./unit-hip-test
     fi 
  fi
fi

if [ "$platform" = "nvcc" ]; then
  
  export HIPSPARSE_LIBRARY_PATH=$current_work_dir/build/lib/src
  export LD_LIBRARY_PATH=$HIPSPARSE_LIBRARY_PATH:$LD_LIBRARY_PATH
  
  cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir
  make package
  make
  echo "${green}HIPSPARSE Build Completed!${reset}"

  if  [ "$testing" = "on" ]; then
       cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler  -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
     set +e
     make -j$working_threads
     ${current_work_dir}/build/test/unit-hip/bin/unit-hip-test
  fi
fi
#if grep --quiet hcsparse ~/.bashrc; then
#  echo 
#else
#  eval "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc"
#fi

#cd $current_work_dir
#exec bash  
