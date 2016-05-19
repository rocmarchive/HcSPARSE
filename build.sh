#!/bin/bash -e
#This script is invoked to install the hcsparse library and test sources

# CHECK FOR COMPILER PATH

if [ ! -z $MCWHCCBUILD ];
then
  if [ -x "$MCWHCCBUILD/compiler/bin/clang++" ];
  then
    cmake_c_compiler="$MCWHCCBUILD/compiler/bin/clang"
    cmake_cxx_compiler="$MCWHCCBUILD/compiler/bin/clang++"
  fi

elif [ -x "/opt/rocm/hcc/bin/clang++" ];
then
  cmake_c_compiler="/opt/rocm/hcc/bin/clang"
  cmake_cxx_compiler="/opt/rocm/hcc/bin/clang++"
else
  echo "Clang compiler not found"
  exit 1
fi

#CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$current_work_dir/build/lib/src
export CLAMP_NOTILECHECK=ON

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
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

set +e
# MAKE BUILD DIR
mkdir $current_work_dir/build
mkdir $current_work_dir/build/test
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

# change to library build
cd $build_dir

# Cmake and make libhcblas: Install hcblas
cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir
make package
make

# Test=OFF (Build library and tests)
if ( [ -z $testing ] ) || ( [ "$testing" = "off" ] ); then
  echo "${green}HCSPARSE Installation Completed!${reset}"
# Test=ON (Build and test the library)
elif ( [ "$testing" = "on" ] ); then
 # Build Tests
   cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
   set +e
   mkdir $current_work_dir/build/test/unit_test/bin/
   mkdir $current_work_dir/build/test/gtest/bin/
   set -e
   make

#  cd $current_work_dir/build/test/gtest/bin/
#  ./unittest
fi

#if grep --quiet hcsparse ~/.bashrc; then
#  echo 
#else
#  eval "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc"
#fi

#cd $current_work_dir
#exec bash  
