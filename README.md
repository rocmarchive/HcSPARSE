# ** HcSPARSE ** #

##Introduction: ##

This repository hosts the HCC implementation of SPARSE subroutines. The following are the sub-routines that are implemented

1. csrmv  : Sparse Matrix - dense Vector multiply (SpM-dV)
2. csrmm  : Sparse Matrix - dense Matrix multiply (SpM-dM)
3. conjugate-gradients : Iterative conjugate gradient solver (CG)
4. biconjugate-gradients-stabilized : Iterative biconjugate gradient stabilized solver (BiCGStab)
5. clsparse-dense2csr : Dense to CSR conversions
6. clsparse-coo2csr : COO to CSR conversions 
7. mm-reader : Functions to read matrix market files in COO or CSR format

##Repository Structure: ##

##Prerequisites: ##
* **OS** : Ubuntu 14.04 LTS
* **Ubuntu Pack**: libc6-dev-i386
   // TODO: Need to add more items 


## Installation Steps:    

### A. HCC Compiler Installation: 
   
**Install HCC compiler debian package:**

 (i) Download the debian package from  [Compiler-Debians](https://multicorewareinc.egnyte.com/dl/TD5IwsNEx3)

 (ii) Install the deb package 
               
                  sudo dpkg -i  hcc-0.8.1544-a9f4d2f-ddba18d-Linux.deb

  Note:
      Ignore clamp-bolt, Bolt is not required for hcRNG.