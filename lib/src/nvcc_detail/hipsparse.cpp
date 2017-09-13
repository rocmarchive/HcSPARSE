#include "hipsparse.h"

#ifdef __cplusplus
extern "C" {
#endif

hipsparseStatus_t hipCudaStatusToHIPStatus(cusparseStatus_t cuStatus) 
{
   switch(cuStatus) 
   {
      case CUSPARSE_STATUS_SUCCESS:
         return HIPSPARSE_STATUS_SUCCESS;
      case CUSPARSE_STATUS_NOT_INITIALIZED:
         return HIPSPARSE_STATUS_NOT_INITIALIZED;
      case CUSPARSE_STATUS_ALLOC_FAILED:
         return HIPSPARSE_STATUS_ALLOC_FAILED;
      case CUSPARSE_STATUS_INVALID_VALUE:
         return HIPSPARSE_STATUS_INVALID_VALUE;
      case CUSPARSE_STATUS_ARCH_MISMATCH:
         return HIPSPARSE_STATUS_ARCH_MISMATCH;
      case CUSPARSE_STATUS_MAPPING_ERROR:
         return HIPSPARSE_STATUS_MAPPING_ERROR;
      case CUSPARSE_STATUS_EXECUTION_FAILED:
         return HIPSPARSE_STATUS_EXECUTION_FAILED;
      case CUSPARSE_STATUS_INTERNAL_ERROR:
         return HIPSPARSE_STATUS_INTERNAL_ERROR;
      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
         return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
      default:
         throw "Unimplemented status";
   }
}

cusparseOperation_t hipHIPOperationToCudaOperation(hipsparseOperation_t op)
{
   switch(op)
   {
      case HIPSPARSE_OPERATION_NON_TRANSPOSE:
         return CUSPARSE_OPERATION_NON_TRANSPOSE;
      case HIPSPARSE_OPERATION_TRANSPOSE:
         return CUSPARSE_OPERATION_TRANSPOSE;
      case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
         return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
      default:
         throw "Invalid Operation Specified";
   }
}

cusparseIndexBase_t hipHIPIndexBaseToCudaIndexBase(hipsparseIndexBase_t idBase)
{
   switch(idBase)
   {
      case HIPSPARSE_INDEX_BASE_ZERO:
         return CUSPARSE_INDEX_BASE_ZERO;
      case HIPSPARSE_INDEX_BASE_ONE:
         return CUSPARSE_INDEX_BASE_ONE;
      default:
         throw "Invalid Index Base Specified";
   }
}

cusparseMatrixType_t hipHIPMatrixTypeToCudaMatrixType(hipsparseMatrixType_t matType)
{
   switch(matType)
   {
      case HIPSPARSE_MATRIX_TYPE_GENERAL:
         return CUSPARSE_MATRIX_TYPE_GENERAL;
      case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
         return CUSPARSE_MATRIX_TYPE_SYMMETRIC;
      case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
         return CUSPARSE_MATRIX_TYPE_HERMITIAN;
      case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
         return CUSPARSE_MATRIX_TYPE_TRIANGULAR;
      default:
         throw "Invalid Matrix Type Specified";
   }
}

cusparseDirection_t hipHIPDirectionToCudaDirection(hipsparseDirection_t dir)
{
   switch(dir)
   {
      case HIPSPARSE_DIRECTION_ROW:
         return CUSPARSE_DIRECTION_ROW;
      case HIPSPARSE_DIRECTION_COLUMN:
         return CUSPARSE_DIRECTION_COLUMN;
      default:
         throw "Invalid Index Base Specified";
   }
}

cusparseAction_t hipHIPActionToCudaAction(hipsparseAction_t act)
{
   switch(act)
   { 
     case HIPSPARSE_ACTION_SYMBOLIC:
        return CUSPARSE_ACTION_SYMBOLIC;
     case HIPSPARSE_ACTION_NUMERIC:
        return CUSPARSE_ACTION_NUMERIC;
     default:
        throw "Invalid Action Specified";
   }
}

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t *handle)
{
   return hipCudaStatusToHIPStatus(cusparseCreate(handle));
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
   return hipCudaStatusToHIPStatus(cusparseDestroy(handle));
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t *descrA)
{
   return hipCudaStatusToHIPStatus(cusparseCreateMatDescr(descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
   return hipCudaStatusToHIPStatus(cusparseDestroyMatDescr(descrA));
}


hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA,
                                      hipsparseMatrixType_t type)
{
  return hipCudaStatusToHIPStatus(cusparseSetMatType(descrA,
                                      hipHIPMatrixTypeToCudaMatrixType(type)));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA,
                                           hipsparseIndexBase_t base)
{
  return hipCudaStatusToHIPStatus(cusparseSetMatIndexBase(descrA,
                                      hipHIPIndexBaseToCudaIndexBase(base)));
}

// Sparse L1 BLAS operations

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz,
                              const float           *xVal,
                              const int *xInd, const float           *y,
                              float           *resultDevHostPtr,
                              hipsparseIndexBase_t idxBase){

  return hipCudaStatusToHIPStatus(cusparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr,
                                                hipHIPIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t handle, int nnz,
                              const double           *xVal,
                              const int *xInd, const double  *y,
                              double           *resultDevHostPtr,
                              hipsparseIndexBase_t idxBase){

  return hipCudaStatusToHIPStatus(cusparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr,
                                                hipHIPIndexBaseToCudaIndexBase(idxBase)));
}


//Sparse L2 BLAS operations

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const float           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const float           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const float           *x, const float           *beta, 
                                  float           *y) {

    return hipCudaStatusToHIPStatus(cusparseScsrmv(handle, hipHIPOperationToCudaOperation(transA),
                                                   m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                                   csrColIndA, x, beta, y));
}

hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const double           *x, const double           *beta, 
                                  double           *y) {

    return hipCudaStatusToHIPStatus(cusparseDcsrmv(handle, hipHIPOperationToCudaOperation(transA),
                                                   m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                                   csrColIndA, x, beta, y));
}


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
                                  const float *beta, float *C, int ldc) {

    return hipCudaStatusToHIPStatus(cusparseScsrmm(handle, hipHIPOperationToCudaOperation(transA),
                                                   m, n, k, nnz, alpha, descrA, csrValA, 
                                                   csrRowPtrA, csrColIndA, B, 
                                                   ldb, beta, C, ldc));
}

hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t handle, 
                                  hipsparseOperation_t transA, 
                                  int m, int n, int k, int nnz, 
                                  const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double             *csrValA, 
                                  const int             *csrRowPtrA, 
                                  const int             *csrColIndA,
                                  const double *B,             int ldb,
                                  const double *beta, double *C, int ldc) {

    return hipCudaStatusToHIPStatus(cusparseDcsrmm(handle, hipHIPOperationToCudaOperation(transA),
                                                   m, n, k, nnz, alpha, descrA, csrValA, 
                                                   csrRowPtrA, csrColIndA, B, 
                                                   ldb, beta, C, ldc));
}

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
                                    int *csrColIndC ) { 

    return hipCudaStatusToHIPStatus(cusparseScsrgemm(handle, 
                                                     hipHIPOperationToCudaOperation(transA),
                                                     hipHIPOperationToCudaOperation(transB), m, 
                                                     n, k, descrA, nnzA, csrValA,
                                                     csrRowPtrA, csrColIndA,
                                                     descrB, nnzB, csrValB, 
                                                     csrRowPtrB, csrColIndB,
                                                     descrC, csrValC,
                                                     csrRowPtrC, csrColIndC));
}

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
                                    int *csrColIndC ) { 

    return hipCudaStatusToHIPStatus(cusparseDcsrgemm(handle, 
                                                     hipHIPOperationToCudaOperation(transA),
                                                     hipHIPOperationToCudaOperation(transB), m, 
                                                     n, k, descrA, nnzA, csrValA,
                                                     csrRowPtrA, csrColIndA,
                                                     descrB, nnzB, csrValB, 
                                                     csrRowPtrB, csrColIndB,
                                                     descrC, csrValC,
                                                     csrRowPtrC, csrColIndC));
}

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
                                    int *csrColIndC) {

   return hipCudaStatusToHIPStatus(cusparseScsrgeam(handle, m, n, alpha, descrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, beta, descrB, nnzB,
                                                        csrValB, csrRowPtrB, csrColIndB,
                                                        descrC, csrValC, csrRowPtrC,
                                                        csrColIndC));

}

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
                                    int *csrColIndC) {

   return hipCudaStatusToHIPStatus(cusparseDcsrgeam(handle, m, n, alpha, descrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, beta, descrB, nnzB,
                                                        csrValB, csrRowPtrB, csrColIndB,
                                                        descrC, csrValC, csrRowPtrC,
                                                        csrColIndC));

}


//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float           *A, 
                                                    int lda, const int *nnzPerRow, 
                                                    float           *csrValA, 
                                                    int *csrRowPtrA, int *csrColIndA) {
    
    return hipCudaStatusToHIPStatus(cusparseSdense2csr(handle, m, n, descrA, A, 
                                                           lda, nnzPerRow, csrValA, 
                                                           csrRowPtrA, csrColIndA) ); 
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double           *A, 
                                                    int lda, const int *nnzPerRow, 
                                                    double           *csrValA, 
                                                    int *csrRowPtrA, int *csrColIndA) {
    
    return hipCudaStatusToHIPStatus(cusparseDdense2csr(handle, m, n, descrA, A, 
                                                           lda, nnzPerRow, csrValA, 
                                                           csrRowPtrA, csrColIndA) ); 
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda){

    return hipCudaStatusToHIPStatus(cusparseScsr2dense(handle, m, n, descrA,
                                                            csrValA, csrRowPtrA,
                                                            csrColIndA, A, lda));
}

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda){

    return hipCudaStatusToHIPStatus(cusparseDcsr2dense(handle, m, n, descrA,
                                                            csrValA, csrRowPtrA,
                                                            csrColIndA, A, lda));
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase){

    return hipCudaStatusToHIPStatus(cusparseXcoo2csr(handle, cooRowIndA, nnz, m, csrRowPtrA,
                                                     hipHIPIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase){

  return hipCudaStatusToHIPStatus(cusparseXcsr2coo(handle, csrRowPtrA, nnz, m, cooRowIndA,
                                                   hipHIPIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m,
                              int n, const hipsparseMatDescr_t descrA,
                              const float           *A, int lda,
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

  return hipCudaStatusToHIPStatus(cusparseSnnz(handle, hipHIPDirectionToCudaDirection(dirA),
                                               m, n, descrA, A, lda,
                                               nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m,
                              int n, const hipsparseMatDescr_t descrA,
                              const double           *A, int lda,
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

  return hipCudaStatusToHIPStatus(cusparseDnnz(handle, hipHIPDirectionToCudaDirection(dirA),
                                               m, n, descrA, A, lda,
                                               nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n,
                                      const hipsparseMatDescr_t descrA,
                                      const float           *cscValA,
                                      const int *cscRowIndA, const int *cscColPtrA,
                                      float           *A, int lda){

  return hipCudaStatusToHIPStatus(cusparseScsc2dense(handle, m, n, descrA,
                                                     cscValA, cscRowIndA, 
                                                     cscColPtrA, A, lda));
}

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t handle, int m, int n,
                                      const hipsparseMatDescr_t descrA,
                                      const double  *cscValA,
                                      const int *cscRowIndA, const int *cscColPtrA,
                                      double           *A, int lda){

  return hipCudaStatusToHIPStatus(cusparseDcsc2dense(handle, m, n, descrA,
                                                     cscValA, cscRowIndA, 
                                                     cscColPtrA, A, lda));
}

hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n,
                                      const hipsparseMatDescr_t descrA,
                                      const float           *A,
                                      int lda, const int *nnzPerCol,
                                      float           *cscValA,
                                      int *cscRowIndA, int *cscColPtrA){

  return hipCudaStatusToHIPStatus(cusparseSdense2csc(handle, m, n, descrA,
                                                     A, lda, nnzPerCol, cscValA,
                                                     cscRowIndA, cscColPtrA));
}

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t handle, int m, int n,
                                      const hipsparseMatDescr_t descrA,
                                      const double           *A,
                                      int lda, const int *nnzPerCol,
                                      double          *cscValA,
                                      int *cscRowIndA, int *cscColPtrA){

  return hipCudaStatusToHIPStatus(cusparseDdense2csc(handle, m, n, descrA,
                                                     A, lda, nnzPerCol, cscValA,
                                                     cscRowIndA, cscColPtrA));
}

hipsparseStatus_t 
hipsparseScsr2csc(hipsparseHandle_t handle, int m, int n, int nnz,
                 const float *csrVal, const int *csrRowPtr, 
                 const int *csrColInd, float           *cscVal,
                 int *cscRowInd, int *cscColPtr, 
                 hipsparseAction_t copyValues, 
                 hipsparseIndexBase_t idxBase)

{

  return hipCudaStatusToHIPStatus(cusparseScsr2csc(handle, m, n, nnz, csrVal,
                                                   csrRowPtr, csrColInd, cscVal,
                                                   cscRowInd, cscColPtr,
                                                   hipHIPActionToCudaAction(copyValues),
                                                   hipHIPIndexBaseToCudaIndexBase(idxBase)));

}

hipsparseStatus_t 
hipsparseDcsr2csc(hipsparseHandle_t handle, int m, int n, int nnz,
                 const double *csrVal, const int *csrRowPtr, 
                 const int *csrColInd, double           *cscVal,
                 int *cscRowInd, int *cscColPtr, 
                 hipsparseAction_t copyValues, 
                 hipsparseIndexBase_t idxBase)

{

  return hipCudaStatusToHIPStatus(cusparseDcsr2csc(handle, m, n, nnz, csrVal,
                                                   csrRowPtr, csrColInd, cscVal,
                                                   cscRowInd, cscColPtr, 
                                                   hipHIPActionToCudaAction(copyValues),
                                                   hipHIPIndexBaseToCudaIndexBase(idxBase)));

}
#ifdef __cplusplus
}
#endif
