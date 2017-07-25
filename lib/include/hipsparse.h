/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified 
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//!  This is the master include file for hipsparse, wrapping around hcsparse and cusparse "version 1"
//

#pragma once

enum hipsparseStatus_t {
  HIPSPARSE_STATUS_SUCCESS,          // Function succeeds
  HIPSPARSE_STATUS_NOT_INITIALIZED,  // HIPSPARSE library not initialized
  HIPSPARSE_STATUS_ALLOC_FAILED,     // resource allocation failed
  HIPSPARSE_STATUS_INVALID_VALUE,    // unsupported numerical value was passed to function
  HIPSPARSE_STATUS_MAPPING_ERROR,    // access to GPU memory space failed
  HIPSPARSE_STATUS_EXECUTION_FAILED, // GPU program failed to execute
  HIPSPARSE_STATUS_INTERNAL_ERROR,    // an internal HIPSPARSE operation failed
  HIPSPARSE_STATUS_NOT_SUPPORTED     // cuSPARSE supports this, but not hcSPARSE
};

enum hipsparseIndexBase_t {
  HIPSPARSE_INDEX_BASE_ZERO,  // the base index is zero.
  HIPSPARSE_INDEX_BASE_ONE
};

enum hipsparseMatrixType_t {
  HIPSPARSE_MATRIX_TYPE_GENERAL,    // the matrix is general.
  HIPSPARSE_MATRIX_TYPE_SYMMETRIC,  // the matrix is symmetric.
  HIPSPARSE_MATRIX_TYPE_HERMITIAN,  // the matrix is hermitian.
  HIPSPARSE_MATRIX_TYPE_TRIANGULAR  // the matrix is triangular.
};

enum hipsparseOperation_t {
  HIPSPARSE_OPERATION_NON_TRANSPOSE,
  HIPSPARSE_OPERATION_TRANSPOSE,
  HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
};

enum hipsparseDirection_t {
  HIPSPARSE_DIRECTION_ROW,
  HIPSPARSE_DIRECTION_COLUMN
};

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:

#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__)
#include <hcc_detail/hip_sparse.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_sparse.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


