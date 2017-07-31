#!/bin/bash -e
#This script is invoked to install the hcsparse library and test sources

# getconf _NPROCESSORS_ONLN
working_threads=8

# CHECK FOR COMPILER PATH
if [ ! -z $HCC_HOME ]
then
  platform="hcc"
  if [ -x "$HCC_HOME/bin/clang++" ]
  then
    cmake_c_compiler="$HCC_HOME/bin/clang"
    cmake_cxx_compiler="$HCC_HOME/bin/clang++"
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

#CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

if [ ! -z $HIP_PATH ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIP_PATH/lib
else
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib
fi

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`
copt="-O3"
verbose=""
install=0

# Help menu
print_help() {
cat <<-HELP
=============================================================================================================================
This script is invoked to build and install hcSPARSE library and test sources. Please provide the following arguments:

  ${green}--test${reset}    Test to enable the library testing. 
  ${green}--debug${reset}    Compile with debug info (-g)
  ${green}--verbose${reset}  Run make with VERBOSE=1
  ${green}--install${reset}  Install the shared library and include the header files under /opt/rocm/hcblas  Requires sudo perms.
  ${green}--examples${reset} To build and run the example files in examples folder (on/off) (ONLY SUPPORTED ON AMD PLATFORM)
  
=============================================================================================================================
HELP
exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --test=*)
      testing="${1#*=}"
      ;;
    --debug)
      copt="-g"
      ;;
    --verbose)
      verbose="VERBOSE=1"
      ;;
    --install)
      install="1"
      ;;
    --examples=*)
      examples="${1#*=}"
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
mkdir -p $current_work_dir/build
mkdir -p $current_work_dir/build/packaging
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

# change to library build
cd $build_dir

if [ "$platform" = "hcc" ]; then
    #default case: Of course there are Compulsory waits on certain kernels 
    cmake -DCMAKE_C_COMPILER=$cmake_c_compiler  -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS="$copt -fPIC" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir

  make -j$working_threads package $verbose
  make -j$working_threads $verbose

  export HCSPARSE_LIBRARY_PATH=$current_work_dir/build/lib/src
  export LD_LIBRARY_PATH=$HCSPARSE_LIBRARY_PATH:$LD_LIBRARY_PATH

   if [ "$install" = "1" ]; then
     sudo make -j$working_threads install
   fi
   cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir/packaging/
   
  echo "${green}hcSPARSE Build Completed!${reset}"

# Test=ON (Build and test the library)
  if ( [ "$testing" = "on" ] ) || ( [ "$testing" = "basic" ] ); then
# Build Tests
     set +e
     mkdir -p $current_work_dir/build/test
     mkdir -p $current_work_dir/build/test/unit_test/bin/
     mkdir -p $current_work_dir/build/test/gtest/bin/
     mkdir -p $current_work_dir/build/test/unit-hip/bin/
     mkdir -p $current_work_dir/build/test/unit-API/bin/
     set -e
     
     cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
     make -j$working_threads

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

#EXAMPLES
#Invoke examples script if --examples=on
  if [ "$examples" = "on" ]; then
    printf "* EXAMPLES *\n"
    printf "************\n"
    chmod +x $current_work_dir/examples/build.sh
    cd $current_work_dir/examples/
    ./build.sh
  fi

if [ "$platform" = "nvcc" ]; then
  
  export HIPSPARSE_LIBRARY_PATH=$current_work_dir/build/lib/src
  export LD_LIBRARY_PATH=$HIPSPARSE_LIBRARY_PATH:$LD_LIBRARY_PATH
  
  cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir

  make -j$working_threads package $verbose
  make -j$working_threads $verbose

  if [ "$install" = "1" ]; then
    sudo -j$working_threads make install
  fi
  cd $build_dir/packaging/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=/opt/rocm/hcsparse $current_work_dir/packaging/ 

  echo "${green}hipSPARSE Build Completed!${reset}"

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
