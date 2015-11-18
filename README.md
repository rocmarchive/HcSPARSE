# ** HcSPARSE ** #

##Introduction: ##

This repository hosts the C++ AMP implementation of SPARSE subroutines. The following are the sub-routines that are implemented

1. csrmv  : Sparse Matrix - dense Vector multiply (SpM-dV)
2. csrmm  : Sparse Matrix - dense Matrix multiply (SpM-dM)
3. conjugate-gradients : Iterative conjugate gradient solver (CG)
4. biconjugate-gradients-stabilized : Iterative biconjugate gradient stabilized solver (BiCGStab)
5. clsparse-dense2csr : Dense to CSR conversions
6. clsparse-coo2csr : COO to CSR conversions 
7. mm-reader : Functions to read matrix market files in COO or CSR format

##Repository Structure: ##

##Prerequisites: ##
* **dGPU**:  AMD firepro S9150
* **OS** : Ubuntu 14.04 LTS
* **Ubuntu Pack**: libc6-dev-i386
* **AMD APP SDK** : Ver 2.9.1 launched on 18/8/2014 from [here](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* **AMD Driver installer**: amd-driver-installer-14.301.1001-x86.x86_64

## Installation Steps:    

### A. C++ AMP Compiler Installation: 
    
** Build from source **

  To build the compiler from source follow the steps given below,
 
  Make sure the parent directory chosen is say ~/ or any other folder of your choice. Lets take ~/ as an example

  (a) Prepare a directory for work space

       * mkdir ~/mcw_cppamp

       * cd ~/mcw_cppamp 
   
       * git clone https://bitbucket.org/multicoreware/cppamp-driver-ng.git src

       * cd ~/mcw_cppamp/src/

       * git checkout origin/torch-specific

  (b) Create a build directory and configure using CMake.

       * mkdir ~/mcw_cppamp/build

       * cd ~/mcw_cppamp/build

       * export CLAMP_NOTILECHECK=ON
       
       * cmake ../src -DCMAKE_BUILD_TYPE=Release -DCXXAMP_ENABLE_BOLT=ON -DOPENCL_HEADER_DIR=<path to SDK's OpenCL headers> -DOPENCL_LIBRARY_DIR=<path to SDK's OpenCL library>, 

       * for example cmake ../src -DCMAKE_BUILD_TYPE=Release -DCXXAMP_ENABLE_BOLT=ON  -DOPENCL_HEADER_DIR=/opt/AMDAPPSDK-2.9-1/include/CL/ -DOPENCL_LIBRARY_DIR=/opt/AMDAPPSDK-2.9-1/lib/x86_64/

  (c) Build AMP

       * cd ~/mcw_cppamp/build

       * make [-j#] world && make          (# is the number of parallel builds. Generally it is # of CPU cores)

       * For example: make -j8 world && make

With this the C++ AMP Compiler installation is complete.

