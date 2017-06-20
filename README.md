## HcSPARSE ##

## A. Introduction: ##

This repository hosts the HCC implementation of linear algebraic routines for sparse matrices/vectors (HcSparse), on AMD devices. To know what HCC compiler features, refer [here](https://github.com/RadeonOpenCompute/hcc). 

Following list the routines that are currently supported by HcSparse.

   1. Level 1 BLAS Routines:
      * axpy : Product of Sparse Vector with constant and followed by addition of dense Vector.
      * dot : Dot product of Sparse Vector with dense Vector
   2. Level 2 BLAS Routines:
      * csrmv  : Sparse Matrix - dense Vector multiply (SpM-dV) 
      * csrmm  : Sparse Matrix - dense Matrix multiply (SpM-dM)
   3. Level 3 BLAS Routines:
      * csrgemm : Sparse Matrix - Sparse Matrix multiply 
      * csrgeam : Sparse Matrix - Sparse Matrix addition
   4. Conversion Routines:
      * dense2csr : Dense to CSR conversions
      * csr2dense : CSR to Dense conversions
      * coo2csr : COO to CSR conversions 
      * dense2csc : Dense to CSC conversions
      * csc2dense : CSC to Dense conversions
   Input Reader functions:
      * mm-reader : Functions to read matrix market files in COO or CSR format

## B. Key Features: ##

   * Supports COO, CSR, CSC, Dense matrix formats.
   * Level 1 Routines for sparse vector with dense formats
   * Level 2 routines for sparse matrix with dense vector/matrix formats
   * Level 2 routines for sparse matrix with sparse matrix operations.
   * Conversion routines that supports conversion between different matrix formats.
   * conjugate-gradients : iterative conjugate gradient solver (CG)
   * biconjugate-gradients-stabilized : Iterative biconjugate gradient stabilized solver (BiCGStab)

## C. Prerequisites ##

* Refer Prerequisites section [here](https://github.com/ROCmSoftwarePlatform/HcSPARSE/wiki/Prerequisites)

## D. Tested Environment so far 

* Refer Tested environments enumerated [here](https://github.com/ROCmSoftwarePlatform/HcSPARSE/wiki/Tested-Environments)

## E. Installation  

* Follow installation steps as described [here](https://github.com/ROCmSoftwarePlatform/HcSPARSE/wiki/Installation)

## F. Unit testing

* Follow testing procedures as explained [here](https://github.com/ROCmSoftwarePlatform/HcSPARSE/wiki/Unit-testing)
