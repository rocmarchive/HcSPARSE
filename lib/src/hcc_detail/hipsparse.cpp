#include "hipsparse.h"

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
   return hipHCSPARSEStatusToHIPStatus(hcsparseDestroyMatDescr(&descrA));
}

// Not used for CNTK requirement
// Will enable it in future
#if 0
//Sparse L2 BLAS operations

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const float           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const float           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const float           *x, const float           *beta, 
                                  float           *y) {

   hcsparseCsrMatrix gCsrMat;
   hcdenseVector gX;
   hcdenseVector gY;
   hcsparseScalar gAlpha;
   hcsparseScalar gBeta;

   int deviceId;
   hipError_t err;
   hipsparseStatus_t retval = HIPSPARSE_STATUS_SUCCESS;

   err = hipGetDevice(&deviceId);
   hc::accelerator_view *av;

   err = hipHccGetAcceleratorView(hipStreamDefault, &av);
   if (err == hipSuccess) {
     hcsparseControl control(*av);
   }
   else{
     return HIPSPARSE_STATUS_EXECUTION_FAILED;
   }
   
   array_view<float> dev_X(n, x);
   array_view<float> dev_Y(m, y);
   array_view<float> dev_alpha(1, alpha);
   array_view<float> dev_beta(1, beta);

   hcsparseSetup();
   hcsparseInitCsrMatrix(&gCsrMat);
   hcsparseInitScalar(&gAlpha);
   hcsparseInitScalar(&gBeta);
   hcsparseInitVector(&gX);
   hcsparseInitVector(&gY);

   gAlpha.value = &dev_alpha;
   gBeta.value = &dev_beta;
   gX.values = &dev_X;
   gY.values = &dev_Y;

   gAlpha.offValue = 0;
   gBeta.offValue = 0;
   gX.offValues = 0;
   gY.offValues = 0;

   gX.num_values = n;
   gY.num_values = m;

   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   array_view<float> av_values(nnz, csrValA);
   array_view<int> av_rowOff(m+1, csrRowPtrA);
   array_view<int> av_colIndices(nnz, csrColIndA);

   gCsrMat.values = &av_values;
   gCsrMat.rowOffsets = &av_rowOff;
   gCsrMat.colIndices = &av_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmv( &gAlpha, gCsrMat, &gx, &gbeta,
                                                        &gy, &control));
}


hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, 
                                  int m, int n, int nnz, const double           *alpha, 
                                  const hipsparseMatDescr_t descrA, 
                                  const double           *csrValA, 
                                  const int *csrRowPtrA, const int *csrColIndA,
                                  const double           *x, const double           *beta, 
                                  double           *y) {

   hcsparseCsrMatrix gCsrMat;
   hcdenseVector gX;
   hcdenseVector gY;
   hcsparseScalar gAlpha;
   hcsparseScalar gBeta;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view;
   hcsparseControl control(accl_view);

   array_view<double> dev_X(n, x);
   array_view<double> dev_Y(m, y);
   array_view<double> dev_alpha(1, alpha);
   array_view<double> dev_beta(1, beta);

   hcsparseSetup();
   hcsparseInitCsrMatrix(&gCsrMat);
   hcsparseInitScalar(&gAlpha);
   hcsparseInitScalar(&gBeta);
   hcsparseInitVector(&gX);
   hcsparseInitVector(&gY);

   gAlpha.value = &dev_alpha;
   gBeta.value = &dev_beta;
   gX.values = &dev_X;
   gY.values = &dev_Y;

   gAlpha.offValue = 0;
   gBeta.offValue = 0;
   gX.offValues = 0;
   gY.offValues = 0;

   gX.num_values = n;
   gY.num_values = m;

   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   array_view<double> av_values(nnz, csrValA);
   array_view<int> av_rowOff(m+1, csrRowPtrA);
   array_view<int> av_colIndices(nnz, csrColIndA);

   gCsrMat.values = &av_values;
   gCsrMat.rowOffsets = &av_rowOff;
   gCsrMat.colIndices = &av_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrmv( &gAlpha, gCsrMat, &gx, &gbeta,
                                                        &gy, &control));

}
#endif

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


   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmm(handle,transA, m, n, k, nnz,
                                                      alpha, descrA, csrValA, csrRowPtrA,
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

   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsrmm(handle, transA, m, n, k, nnz,
                                                      alpha, descrA, csrValA, csrRowPtrA,
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

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrgemm(handle, transA,
                                                          transB, m, n, k, descrA,
                                                          nnzA, csrValA, csrRowPtrA,
                                                          csrColIndA, descrB, nnzB,
                                                          csrValB, csrRowPtrB,
                                                          csrColIndB, descrC,
                                                          csrValC, csrRowPtrC,
                                                          csrColIndC));
}


//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const float           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      float           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   return hipHCSPARSEStatusToHIPStatus(hcsparseSdense2csr( handle, m, n, descrA, 
                                                           A, lda, nnzPerRow, csrValA,
                                                           csrRowPtrA, csrColIndA)); 
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const double           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      double           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   return hipHCSPARSEStatusToHIPStatus(hcsparseDdense2csr( handle, m, n, descrA, 
                                                           A, lda, nnzPerRow, csrValA,
                                                           csrRowPtrA, csrColIndA)); 
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda){

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsr2dense( handle, m, n,
                                                          descrA, csrValA, csrRowPtrA,
                                                          csrColIndA, A, lda));
}


hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda){

   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsr2dense( handle, m, n,
                                                          descrA, csrValA, csrRowPtrA,
                                                          csrColIndA, A, lda));
}


hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase){

   return hipHCSPARSEStatusToHIPStatus(hcsparseXcoo2csr(handle, cooRowIndA, nnz,
                                                        m, csrRowPtrA, idxBase));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase){

  return hipHCSPARSEStatusToHIPStatus(hcsparseXcsr2coo(handle, csrRowPtrA, nnz,
                                                       m, cooRowIndA, idxBase));
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const float           *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

  return hipHCSPARSEStatusToHIPStatus(hcsparseSnnz(handle, dirA, m, n, descrA, A, lda,
                                      nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, 
                              int n, const hipsparseMatDescr_t descrA, 
                              const double *A, int lda, 
                              int *nnzPerRowColumn, int *nnzTotalDevHostPtr){

  return hipHCSPARSEStatusToHIPStatus(hcsparseDnnz(handle, dirA, m, n, descrA, A, lda,
                                      nnzPerRowColumn, nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz, 
                              const float           *xVal, 
                              const int *xInd, const float           *y, 
                              float           *resultDevHostPtr, 
                              hipsparseIndexBase_t idxBase){

  return hipHCSPARSEStatusToHIPStatus(hcsparseSdoti(handle, nnz, xVal, xInd, y, 
                                                    resultDevHostPtr, idxBase));
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              float           *A, int lda){

  return hipHCSPARSEStatusToHIPStatus(hcsparseScsc2dense(handle, m, n, descrA, 
                                                         cscValA, cscRowIndA, 
                                                         cscColPtrA, A, lda));
}

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double *cscValA, 
                              const int *cscRowIndA, const int *cscColPtrA,
                              double    *A, int lda){

  return hipHCSPARSEStatusToHIPStatus(hcsparseDcsc2dense(handle, m, n, descrA, 
                                                         cscValA, cscRowIndA, 
                                                         cscColPtrA, A, lda));
}


hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const float           *A, 
                              int lda, const int *nnzPerCol, 
                              float           *cscValA, 
                              int *cscRowIndA, int *cscColPtrA){

  return hipHCSPARSEStatusToHIPStatus(hcsparseSdense2csc(handle, m, n, descrA, A, 
                                                         lda, nnzPerCol, cscValA, 
                                                         cscRowIndA, cscColPtrA));
}

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t handle, int m, int n, 
                              const hipsparseMatDescr_t descrA, 
                              const double *A, 
                              int lda, const int *nnzPerCol, 
                              double *cscValA, 
                              int *cscRowIndA, int *cscColPtrA){

  return hipHCSPARSEStatusToHIPStatus(hcsparseDdense2csc(handle, m, n, descrA, A, 
                                                         lda, nnzPerCol, cscValA, 
                                                         cscRowIndA, cscColPtrA));
}

#ifdef __cplusplus
}
#endif

