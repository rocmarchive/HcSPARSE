# ** HcSPARSE ** #

##Introduction: ##

This repository hosts the HCC implementation of SPARSE subroutines. The following are the sub-routines that are implemented

1. csrmv  : Sparse Matrix - dense Vector multiply (SpM-dV)
2. csrmm  : Sparse Matrix - dense Matrix multiply (SpM-dM)
3. conjugate-gradients : Iterative conjugate gradient solver (CG)
4. biconjugate-gradients-stabilized : Iterative biconjugate gradient stabilized solver (BiCGStab)
5. dense2csr : Dense to CSR conversions
6. csr2dense : CSR to Dense conversions
7. coo2csr : COO to CSR conversions 
8. mm-reader : Functions to read matrix market files in COO or CSR format

##Prerequisites: ##

**Hardware Requirements:**

* CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU 


**GPU SDK and driver Requirements:**

* AMD R9 Fury X, R9 Fur, R9 Nano + Boltzmann driver
* AMD APU Kaveri or Carrizo + HSA driver

**System software requirements:**

* Ubuntu 14.04 trusty
* GCC 4.6 and later


## Installation Steps:    

### A. HCC Compiler Installation: 
   
**Install HCC compiler debian package:**

 (i) Download the debian package from  [Compiler-Debians](https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.8.16024-6617a6a-a152a36-5a1009a-Linux.deb)

 (ii) Install the deb package 
               
                  sudo dpkg -i  hcc-0.8.16024-6617a6a-a152a36-5a1009a-Linux.deb
                  
   Note: The hcc compiler binaries and libraries gets installed under /opt/hcc path
   
** Build and Install HcSparse: **
 
 (i) Clone the hcsparse repo 
 
                   git clone https://bitbucket.org/multicoreware/hcsparse.git
                   
 (ii) Install hcsparse library
                    
                    run ./install.sh
                    
 (iii) Build and run test
 
                    run ./install.sh --test=on  (this will build and run the gtest)