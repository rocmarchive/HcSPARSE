#include "hipsparse.h"

#ifdef __cplusplus
extern "C" {
#endif

hipsparseStatus_t hipCUSPARSEStatusToHIPStatus(cusparseStatus_t cuStatus) 
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
      case CUSPARSE_STATUS_MAPPING_ERROR:
         return HIPSPARSE_STATUS_MAPPING_ERROR;
      case CUSPARSE_STATUS_EXECUTION_FAILED:
         return HIPSPARSE_STATUS_EXECUTION_FAILED;
      case CUSPARSE_STATUS_INTERNAL_ERROR:
         return HIPSPARSE_STATUS_INTERNAL_ERROR;
      case CUSPARSE_STATUS_NOT_SUPPORTED:
         return HIPSPARSE_STATUS_NOT_SUPPORTED;
      default:
         throw "Unimplemented status";
   }
}

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t *handle)
{
   return hipCUSPARSEStatusToHIPStatus(cusparseCreate(handle));
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
   return hipCUSPARSEStatusToHIPStatus(cusparseDestroy(handle));
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t *descrA)
{
   return hipCUSPARSEStatusToHIPStatus(cusparseCreateMatDescr(descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
   return hipCUSPARSEStatusToHIPStatus(cusparseDestroyMatDescr(descrA));
}

//Sparse L2 BLAS operations

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const float           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const float           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const float           *x, const float           *beta, 
                                  float           *y) {

    return hipCUSPARSEStatusToHIPStatus(cusparseScsrmv(handle, transA, m, n, nnz, *alpha, 
                                                       descrA, csrValA, csrRowPtrA,
                                                       csrColIndA, x, beta, y));
}

hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const double           *x, const double           *beta, 
                                  double           *y) {

    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrmv(handle, transA, m, n, nnz, *alpha, 
                                                       descrA, csrValA, csrRowPtrA,
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

    return hipCUSPARSEStatusToHIPStatus(cusparseScsrmm( handle, transA, m, n, k, 
                                                        nnz, alpha, descrA, csrValA, 
                                                        csrRowPtrA, csrColIndA, B, 
                                                        ldb, beta, C, ldc) );
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

    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrmm( handle, transA, m, n, k, 
                                                        nnz, alpha, descrA, csrValA, 
                                                        csrRowPtrA, csrColIndA, B, 
                                                        ldb, beta, C, ldc) );
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

    return hipCUSPARSEStatusToHIPStatus(cusparseScsrgemm(handle, transA, transB, m, 
                                                         n, k, descrA, nnzA, csrValA,
                                                         csrRowPtrA, csrColIndA,
                                                         descrB, nnzB, csrValB, 
                                                         csrRowPtrB, csrColIndB,
                                                         descrC, csrValC,
                                                         csrRowPtrC, csrColIndC ) );
}


//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float           *A, 
                                                    int lda, const int *nnzPerRow, 
                                                    float           *csrValA, 
                                                    int *csrRowPtrA, int *csrColIndA) {
    
    return hipCUSPARSEStatusToHIPStatus(cusparseSdense2csr(handle, m, n, descrA, A, 
                                                           lda, nnzPerRow, csrValA, 
                                                           csrRowPtrA, csrColIndA) ); 
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double           *A, 
                                                    int lda, const int *nnzPerRow, 
                                                    double           *csrValA, 
                                                    int *csrRowPtrA, int *csrColIndA) {
    
    return hipCUSPARSEStatusToHIPStatus(cusparseDdense2csr(handle, m, n, descrA, A, 
                                                           lda, nnzPerRow, csrValA, 
                                                           csrRowPtrA, csrColIndA) ); 
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda){

    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2dense(handle, m, n, descrA,
                                                            csrValA, csrRowPtrA,
                                                            csrColIndA, A, lda));
}

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda){

    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2dense(handle, m, n, descrA,
                                                            csrValA, csrRowPtrA,
                                                            csrColIndA, A, lda));
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase){

    return hipCUSPARSEStatusToHIPStatus(cusparseXcoo2csr(handle, cooRowIndA, nnz, m, csrRowPtrA, idxBase));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase){

  return hipCUSPARSEStatusToHIPStatus(cusparseXcsr2coo(handle, csrRowPtrA, nnz, m, cooRowIndA, idxBase) );
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m,
                              int n, const hipsparseMatDescr_t descrA,
                              const float           *A, int lda,
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

  return hipCUSPARSEStatusToHIPStatus(cusparseSnnz(handle, dirA, m, n, descrA, A, lda,
                                                   nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz,
                              const float           *xVal,
                              const int *xInd, const float           *y,
                              float           *resultDevHostPtr,
                              hipsparseIndexBase_t idxBase){

  return hipCUSPARSEStatusToHIPStatus(cusparseSdoti(handle, nnz, xVal, xInd, y,
                                                    resultDevHostPtr, idxBase));
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n,
                              const hipsparseMatDescr_t descrA,
                              const float           *cscValA,
                              const int *cscRowIndA, const int *cscColPtrA,
                              float           *A, int lda){

  return hipCUSPARSEStatusToHIPStatus(cusparseScsc2dense(handle, m, n, descrA,
                                                         cscValA, cscRowIndA, 
                                                         cscColPtrA, A, lda));
}

hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n,
                              const hipsparseMatDescr_t descrA,
                              const float           *A,
                              int lda, const int *nnzPerCol,
                              float           *cscValA,
                              int *cscRowIndA, int *cscColPtrA){

  return hipCUSPARSEStatusToHIPStatus(cusparseSdense2csc(handle, m, n, descrA,
                                                         A, lda, nnzPerCol, cscValA,
                                                         cscRowIndA, cscColPtrA));
}

#ifdef __cplusplus
}
#endif
