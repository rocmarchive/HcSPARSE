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
#pragma once

#include <hip/hip_runtime_api.h>
#include <hip/hip_hcc.h>

//HGSOS for Kalmar leave it as C++, only cuSPARSE needs C linkage.

#ifdef __cplusplus
extern "C" {
#endif

enum hcsparseIndexBase_t:uint32_t;
enum hcsparseOperation_t:uint32_t;
enum hcsparseDirection_t:uint32_t;
enum hcsparseStatus_t:uint32_t;
enum hcsparseFillMode_t:uint32_t;
enum hcsparseDiagType_t:uint32_t;
enum hcsparseMatrixType_t:uint32_t;
enum hcsparseAction_t:uint32_t;

typedef struct hcsparseMatDescr* hipsparseMatDescr_t;
typedef struct hcsparseLibrary* hipsparseHandle_t ;

hipsparseStatus_t hipHCSPARSEStatusToHIPStatus(hcsparseStatus_t hcStatus); 

hcsparseOperation_t hipHIPOperationToHCSPARSEOperation(hipsparseOperation_t op);

hcsparseIndexBase_t hipHIPIndexBaseToHCSPARSEIndexBase(hipsparseIndexBase_t idBase);

hcsparseMatrixType_t hipHIPMatrixTypeToHCSPARSEMatrixType(hipsparseMatrixType_t matType);

hcsparseDirection_t hipHIPDirectionToHCSPARSEDirection(hipsparseDirection_t dir);

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle);

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle);

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId);

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t *descrA);

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA);

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA,
                                      hipsparseMatrixType_t type);

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, 
                                           hipsparseIndexBase_t base);

//Sparse L1 BLAS operations

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz, 
                              const float           *xVal, 
                              const int *xInd, const float           *y, 
                              float           *resultDevHostPtr, 
                              hipsparseIndexBase_t idxBase);

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t handle, int nnz, 
                              const double           *xVal, 
                              const int *xInd, const double *y, 
                              double           *resultDevHostPtr, 
                              hipsparseIndexBase_t idxBase);

//Sparse L2 BLAS operations

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const float           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const float           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const float           *x, const float           *beta, 
                                  float           *y);

hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const double           *x, const double           *beta, 
                                  double           *y);

//Sparse L3 BLAS operations

hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t handle, 
                                                hipsparseOperation_t transA, 
                                                int m, int n, int k, int nnz, 
                                                const float           *alpha, 
                                                const hipsparseMatDescr_t descrA, 
                                                const float             *csrValA, 
                                                const int             *csrRowPtrA, 
                                                const int             *csrColIndA,
                                                const float *B,             int ldb,
                                                const float *beta, float *C, int ldc);

hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t handle, 
                                                hipsparseOperation_t transA, 
                                                int m, int n, int k, int nnz, 
                                                const double           *alpha, 
                                                const hipsparseMatDescr_t descrA, 
                                                const double             *csrValA, 
                                                const int             *csrRowPtrA, 
                                                const int             *csrColIndA,
                                                const double *B,             int ldb,
                                                const double *beta, double *C, int ldc);

// Sparse Extra BLAS operations

hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t handle,
                                       hipsparseOperation_t transA,
                                       hipsparseOperation_t transB,
                                       int m, int n, int k,
                                       const hipsparseMatDescr_t descrA,
                                       const int nnzA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       const int nnzB,
                                       const int *csrRowPtrB,
                                       const int *csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int *csrRowPtrC,
                                       int *nnzTotalDevHostPtr);

hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t handle,
                                    hipsparseOperation_t transA, 
                                    hipsparseOperation_t transB,
                                    int m, int n, int k,
                                    const hipsparseMatDescr_t descrA, 
                                    const int nnzA, const float *csrValA,
                                    const int *csrRowPtrA, 
                                    const int *csrColIndA,
                                    const hipsparseMatDescr_t descrB, 
                                    const int nnzB, const float *csrValB, 
                                    const int *csrRowPtrB, 
                                    const int *csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float *csrValC, const int *csrRowPtrC, 
                                    int *csrColIndC );

hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t handle,
                                    hipsparseOperation_t transA, 
                                    hipsparseOperation_t transB,
                                    int m, int n, int k,
                                    const hipsparseMatDescr_t descrA, 
                                    const int nnzA, const double *csrValA,
                                    const int *csrRowPtrA, 
                                    const int *csrColIndA,
                                    const hipsparseMatDescr_t descrB, 
                                    const int nnzB, const double *csrValB, 
                                    const int *csrRowPtrB, 
                                    const int *csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double *csrValC, const int *csrRowPtrC, 
                                    int *csrColIndC );

hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t handle, 
                                       int m, int n,
                                       const hipsparseMatDescr_t descrA, 
                                       int nnzA,
                                       const int *csrRowPtrA, 
                                       const int *csrColIndA,
                                       const hipsparseMatDescr_t descrB, 
                                       int nnzB,
                                       const int *csrRowPtrB, 
                                       const int *csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int *csrRowPtrC, 
                                       int *nnzTotalDevHostPtr);

hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t handle, 
                                    int m, int n, const float *alpha,
                                    const hipsparseMatDescr_t descrA, 
                                    int nnzA,
                                    const float *csrValA, 
                                    const int *csrRowPtrA, 
                                    const int *csrColIndA,
                                    const float *beta,
                                    const hipsparseMatDescr_t descrB, 
                                    int nnzB,
                                    const float *csrValB, 
                                    const int *csrRowPtrB, 
                                    const int *csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float *csrValC, 
                                    int *csrRowPtrC, 
                                    int *csrColIndC);

hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t handle, 
                                    int m, int n, const double *alpha,
                                    const hipsparseMatDescr_t descrA, 
                                    int nnzA,
                                    const double *csrValA, 
                                    const int *csrRowPtrA, 
                                    const int *csrColIndA,
                                    const double *beta,
                                    const hipsparseMatDescr_t descrB, 
                                    int nnzB,
                                    const double *csrValB, 
                                    const int *csrRowPtrB, 
                                    const int *csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double *csrValC, 
                                    int *csrRowPtrC, 
                                    int *csrColIndC);

//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const float           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      float           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA);

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const double           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      double           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA);

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda);

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda);

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase);

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase);

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              float           *A, int lda);

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              double *A, int lda);

hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *A, 
                              int lda, const int *nnzPerCol, 
                              float           *cscValA, 
                              int *cscRowIndA, int *cscColPtrA);

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double           *A, 
                              int lda, const int *nnzPerCol, 
                              double           *cscValA, 
                              int *cscRowIndA, int *cscColPtrA);

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const float           *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr);

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const double           *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr);

hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t handle, int m, int n, int nnz,
                              const double *csrVal, const int *csrRowPtr,
                              const int *csrColInd, double *cscVal,
                              int *cscRowInd, int *cscColPtr,
                              hipsparseAction_t copyValues,
                              hipsparseIndexBase_t idxBase);

hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t handle, int m, int n, int nnz,
                              const float *csrVal, const int *csrRowPtr,
                              const int *csrColInd, float *cscVal,
                              int *cscRowInd, int *cscColPtr,
                              hipsparseAction_t copyValues,
                              hipsparseIndexBase_t idxBase);

#ifdef __cplusplus
}
#endif


