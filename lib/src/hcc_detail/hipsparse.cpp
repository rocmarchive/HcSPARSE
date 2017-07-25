#include "hipsparse.h"
#include "hcsparse.h"

#ifdef __cplusplus
extern "C" {
#endif

hipsparseStatus_t hipHCSPARSEStatusToHIPStatus(hcsparseStatus_t hcStatus) 
{
   switch(hcStatus)
   {
      case HCSPARSE_STATUS_SUCCESS:
         return HIPSPARSE_STATUS_SUCCESS;
      case HCSPARSE_STATUS_NOT_INITIALIZED:
         return HIPSPARSE_STATUS_NOT_INITIALIZED;
      case HCSPARSE_STATUS_ALLOC_FAILED:
         return HIPSPARSE_STATUS_ALLOC_FAILED;
      case HCSPARSE_STATUS_INVALID_VALUE:
         return HIPSPARSE_STATUS_INVALID_VALUE;
      case HCSPARSE_STATUS_MAPPING_ERROR:
         return HIPSPARSE_STATUS_MAPPING_ERROR;
      case HCSPARSE_STATUS_EXECUTION_FAILED:
         return HIPSPARSE_STATUS_EXECUTION_FAILED;
      case HCSPARSE_STATUS_INTERNAL_ERROR:
         return HIPSPARSE_STATUS_INTERNAL_ERROR;
      default:
         throw "Unimplemented status";
   }
}

hcsparseOperation_t hipHIPOperationToHCSPARSEOperation(hipsparseOperation_t op) 
{
   switch(op)
   {
      case HIPSPARSE_OPERATION_NON_TRANSPOSE:
         return HCSPARSE_OPERATION_NON_TRANSPOSE;
      case HIPSPARSE_OPERATION_TRANSPOSE:
         return HCSPARSE_OPERATION_TRANSPOSE;
      case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
         return HCSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
      default:
         throw "Invalid Operation Specified";
   }
}

hcsparseIndexBase_t hipHIPIndexBaseToHCSPARSEIndexBase(hipsparseIndexBase_t idBase) 
{
   switch(idBase)
   {
      case HIPSPARSE_INDEX_BASE_ZERO:
         return HCSPARSE_INDEX_BASE_ZERO;
      case HIPSPARSE_INDEX_BASE_ONE:
         return HCSPARSE_INDEX_BASE_ONE;
      default:
         throw "Invalid Index Base Specified";
   }
}

hcsparseMatrixType_t hipHIPMatrixTypeToHCSPARSEMatrixType(hipsparseMatrixType_t matType) 
{
   switch(matType)
   {
      case HIPSPARSE_MATRIX_TYPE_GENERAL:
         return HCSPARSE_MATRIX_TYPE_GENERAL;
      case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
         return HCSPARSE_MATRIX_TYPE_SYMMETRIC;
      case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
         return HCSPARSE_MATRIX_TYPE_HERMITIAN;
      case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
         return HCSPARSE_MATRIX_TYPE_TRIANGULAR;
      default:
         throw "Invalid Matrix Type Specified";
   }
}

hcsparseDirection_t hipHIPDirectionToHCSPARSEDirection(hipsparseDirection_t dir) 
{
   switch(dir)
   {
      case HIPSPARSE_DIRECTION_ROW:
         return HCSPARSE_DIRECTION_ROW;
      case HIPSPARSE_DIRECTION_COLUMN:
         return HCSPARSE_DIRECTION_COLUMN;
      default:
         throw "Invalid Index Base Specified";
   }
}

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t *handle)
{
   int deviceId;
   hipError_t err;
   hipsparseStatus_t retval = HIPSPARSE_STATUS_SUCCESS;

   err = hipGetDevice(&deviceId);
   if (err == hipSuccess) {
     hc::accelerator_view *av;
     err = hipHccGetAcceleratorView(hipStreamDefault, &av);
     if (err == hipSuccess) {
       retval = hipHCSPARSEStatusToHIPStatus(hcsparseCreate(&*handle, av));
     } else {
       retval = HIPSPARSE_STATUS_EXECUTION_FAILED;
     }
   }
  return retval;
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
   return hipHCSPARSEStatusToHIPStatus(hcsparseDestroy(&handle));
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t *descrA)
{
   return hipHCSPARSEStatusToHIPStatus(hcsparseCreateMatDescr(descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
   return hipHCSPARSEStatusToHIPStatus(hcsparseDestroyMatDescr(descrA));
}
//Sparse L1 BLAS operations

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz, 
                              const float           *xVal, 
                              const int *xInd, const float           *y, 
                              float           *resultDevHostPtr, 
                              hipsparseIndexBase_t idxBase){

  return hipHCSPARSEStatusToHIPStatus(hcsparseSdoti(handle, nnz, xVal, xInd, y, 
                                                    resultDevHostPtr,
                                                    hipHIPIndexBaseToHCSPARSEIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t handle, int nnz, 
                                 const double    *xVal, 
                                 const int *xInd, const double *y, 
                                 double *resultDevHostPtr, 
                                 hipsparseIndexBase_t idxBase){

  return hipHCSPARSEStatusToHIPStatus(hcsparseDdoti(handle, nnz, xVal, xInd, y, 
                                                    resultDevHostPtr,
                                                    hipHIPIndexBaseToHCSPARSEIndexBase(idxBase)));
}

//Sparse L2 BLAS operations

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const float           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const float           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const float           *x, const float           *beta, 
                                  float           *y) {

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmv(handle, hipHIPOperationToHCSPARSEOperation(transA), m, n,
                                                      nnz, alpha, *hcDescrA,
                                                      csrValA, csrRowPtrA,
                                                      csrColIndA, x, beta, y));
}


hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const double           *x, const double           *beta, 
                                  double           *y) {

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrmv(handle, hipHIPOperationToHCSPARSEOperation(transA), m, n,
                                                      nnz, alpha, *hcDescrA,
                                                      csrValA, csrRowPtrA,
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


   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmm(handle,hipHIPOperationToHCSPARSEOperation(transA), m, n, k, nnz,
                                                      alpha, *hcDescrA, csrValA, csrRowPtrA,
                                                      csrColIndA, B, ldb, beta, C, ldc));
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrmm(handle, hipHIPOperationToHCSPARSEOperation(transA), m, n, k, nnz,
                                                      alpha, *hcDescrA, csrValA, csrRowPtrA,
                                                      csrColIndA, B, ldb, beta, C, ldc));
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   const hcsparseMatDescr_t *hcDescrB = reinterpret_cast<const hcsparseMatDescr_t*>(&descrB); 
   const hcsparseMatDescr_t *hcDescrC = reinterpret_cast<const hcsparseMatDescr_t*>(&descrC); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrgemm(handle, 
                                                        hipHIPOperationToHCSPARSEOperation(transA),
                                                        hipHIPOperationToHCSPARSEOperation(transB),
                                                        m, n, k, *hcDescrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, *hcDescrB, nnzB,
                                                        csrValB, csrRowPtrB,
                                                        csrColIndB, *hcDescrC,
                                                        csrValC, csrRowPtrC,
                                                        csrColIndC));
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   const hcsparseMatDescr_t *hcDescrB = reinterpret_cast<const hcsparseMatDescr_t*>(&descrB); 
   const hcsparseMatDescr_t *hcDescrC = reinterpret_cast<const hcsparseMatDescr_t*>(&descrC); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrgemm(handle, 
                                                        hipHIPOperationToHCSPARSEOperation(transA),
                                                        hipHIPOperationToHCSPARSEOperation(transB),
                                                        m, n, k, *hcDescrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, *hcDescrB, nnzB,
                                                        csrValB, csrRowPtrB,
                                                        csrColIndB, *hcDescrC,
                                                        csrValC, csrRowPtrC,
                                                        csrColIndC));
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   const hcsparseMatDescr_t *hcDescrB = reinterpret_cast<const hcsparseMatDescr_t*>(&descrB); 
   const hcsparseMatDescr_t *hcDescrC = reinterpret_cast<const hcsparseMatDescr_t*>(&descrC); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrgemm(handle,
                                                        hipHIPOperationToHCSPARSEOperation(transA),
                                                        hipHIPOperationToHCSPARSEOperation(transB),
                                                        m, n, k, *hcDescrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, *hcDescrB, nnzB,
                                                        csrValB, csrRowPtrB,
                                                        csrColIndB, *hcDescrC,
                                                        csrValC, csrRowPtrC,
                                                        csrColIndC));
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   const hcsparseMatDescr_t *hcDescrB = reinterpret_cast<const hcsparseMatDescr_t*>(&descrB); 
   const hcsparseMatDescr_t *hcDescrC = reinterpret_cast<const hcsparseMatDescr_t*>(&descrC); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrgeam(handle, m, n, alpha, *hcDescrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, beta, *hcDescrB, nnzB,
                                                        csrValB, csrRowPtrB, csrColIndB,
                                                        *hcDescrC, csrValC, csrRowPtrC,
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

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   const hcsparseMatDescr_t *hcDescrB = reinterpret_cast<const hcsparseMatDescr_t*>(&descrB); 
   const hcsparseMatDescr_t *hcDescrC = reinterpret_cast<const hcsparseMatDescr_t*>(&descrC); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrgeam(handle, m, n, alpha, *hcDescrA,
                                                        nnzA, csrValA, csrRowPtrA,
                                                        csrColIndA, beta, *hcDescrB, nnzB,
                                                        csrValB, csrRowPtrB, csrColIndB,
                                                        *hcDescrC, csrValC, csrRowPtrC,
                                                        csrColIndC));

}

//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const float           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      float           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseSdense2csr( handle, m, n, *hcDescrA, 
                                                           A, lda, nnzPerRow, csrValA,
                                                           csrRowPtrA, csrColIndA)); 
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const double           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      double           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDdense2csr( handle, m, n, *hcDescrA, 
                                                           A, lda, nnzPerRow, csrValA,
                                                           csrRowPtrA, csrColIndA)); 
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseScsr2dense( handle, m, n,
                                                          *hcDescrA, csrValA, csrRowPtrA,
                                                          csrColIndA, A, lda));
}


hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsr2dense( handle, m, n,
                                                          *hcDescrA, csrValA, csrRowPtrA,
                                                          csrColIndA, A, lda));
}


hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase){

   return hipHCSPARSEStatusToHIPStatus(hcsparseXcoo2csr(handle, cooRowIndA, nnz,
                                                        m, csrRowPtrA,
                                                        hipHIPIndexBaseToHCSPARSEIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase){

  return hipHCSPARSEStatusToHIPStatus(hcsparseXcsr2coo(handle, csrRowPtrA, nnz,
                                                       m, cooRowIndA,
                                                       hipHIPIndexBaseToHCSPARSEIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const float           *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseSnnz(handle, hipHIPDirectionToHCSPARSEDirection(dirA),
                                      m, n, *hcDescrA, A, lda,
                                      nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const double *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseDnnz(handle, hipHIPDirectionToHCSPARSEDirection(dirA),
                                                   m, n, *hcDescrA, A, lda,
                                                   nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              float           *A, int lda){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseScsc2dense(handle, m, n, *hcDescrA, 
                                                         cscValA, cscRowIndA, 
                                                         cscColPtrA, A, lda));
}

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              double    *A, int lda){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseDcsc2dense(handle, m, n, *hcDescrA, 
                                                         cscValA, cscRowIndA, 
                                                         cscColPtrA, A, lda));
}


hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *A, 
                              int lda, const int *nnzPerCol, 
                              float           *cscValA, 
                              int *cscRowIndA, int *cscColPtrA){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseSdense2csc(handle, m, n, *hcDescrA, A, 
                                                         lda, nnzPerCol, cscValA, 
                                                         cscRowIndA, cscColPtrA));
}

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double *A, 
                              int lda, const int *nnzPerCol, 
                              double *cscValA, 
                              int *cscRowIndA, int *cscColPtrA){

   const hcsparseMatDescr_t *hcDescrA = reinterpret_cast<const hcsparseMatDescr_t*>(&descrA); 
  return hipHCSPARSEStatusToHIPStatus(hcsparseDdense2csc(handle, m, n, *hcDescrA, A, 
                                                         lda, nnzPerCol, cscValA, 
                                                         cscRowIndA, cscColPtrA));
}

#ifdef __cplusplus
}
#endif

