#include "hipsparse.h"

#ifdef __cplusplus
extern "C" {
#endif

hipsparseStatus_t hipHCSPARSEStatusToHIPStatus(hcsparseStatus_t hcStatus) 
{
   switch(hcStatus)
   {
      case hcsparseSuccess:
         return HIPSPARSE_STATUS_SUCCESS;
      case hcsparseInvalid:
         return HIPSPARSE_STATUS_EXECUTION_FAILED;
      default:
         throw "Unimplemented status";
   }
}

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

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view;
   hcsparseControl control(accl_view);

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

   hcsparseCsrMatrix gCsrMat;
   hcdenseMatrix gX;
   hcdenseMatrix gY;
   hcsparseScalar gAlpha;
   hcsparseScalar gBeta;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view;
   hcsparseControl control(accl_view);

   array_view<float> dev_X(k * n , B);
   array_view<float> dev_Y(m * n , C);
   array_view<float> dev_alpha(1, alpha);
   array_view<float> dev_beta(1, beta);

   hcsparseSetup();
   hcsparseInitCsrMatrix(&gCsrMat);
   hcsparseInitScalar(&gAlpha);
   hcsparseInitScalar(&gBeta);
   hcdenseInitMatrix(&gX);
   hcdenseInitMatrix(&gY);

   gAlpha.value = &dev_alpha;
   gBeta.value = &dev_beta;
   gX.values = &dev_X;
   gY.values = &dev_Y;

   gAlpha.offValue = 0;
   gBeta.offValue = 0;
   gX.offValues = 0;
   gY.offValues = 0;

   gX.num_rows = k;
   gX.num_cols = n;
   gX.lead_dim = ldb;
   gY.num_rows = m;
   gY.num_cols = n;
   gY.lead_dim = ldc;

   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   array_view<float> av_values(nnz, csrValA);
   array_view<int> av_rowOff(m+1, csrRowPtrA);
   array_view<int> av_colIndices(nnz, csrColIndA);

   gCsrMat.values = &av_values;
   gCsrMat.rowOffsets = &av_rowOff;
   gCsrMat.colIndices = &av_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmm( &gAlpha, &gCsrMat, &denseMatB,
                                                        &gBeta, &denseMatC, &control ) );
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

   hcsparseCsrMatrix gCsrMat;
   hcdenseMatrix gX;
   hcdenseMatrix gY;
   hcsparseScalar gAlpha;
   hcsparseScalar gBeta;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view;
   hcsparseControl control(accl_view);

   array_view<double> dev_X(k * n , B);
   array_view<double> dev_Y(m * n , C);
   array_view<double> dev_alpha(1, alpha);
   array_view<double> dev_beta(1, beta);

   hcsparseSetup();
   hcsparseInitCsrMatrix(&gCsrMat);
   hcsparseInitScalar(&gAlpha);
   hcsparseInitScalar(&gBeta);
   hcdenseInitMatrix(&gX);
   hcdenseInitMatrix(&gY);

   gAlpha.value = &dev_alpha;
   gBeta.value = &dev_beta;
   gX.values = &dev_X;
   gY.values = &dev_Y;

   gAlpha.offValue = 0;
   gBeta.offValue = 0;
   gX.offValues = 0;
   gY.offValues = 0;

   gX.num_rows = k;
   gX.num_cols = n;
   gX.lead_dim = ldb;
   gY.num_rows = m;
   gY.num_cols = n;
   gY.lead_dim = ldc;

   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   array_view<double> av_values(nnz, csrValA);
   array_view<int> av_rowOff(m+1, csrRowPtrA);
   array_view<int> av_colIndices(nnz, csrColIndA);

   gCsrMat.values = &av_values;
   gCsrMat.rowOffsets = &av_rowOff;
   gCsrMat.colIndices = &av_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrmm( &gAlpha, &gCsrMat, &denseMatB,
                                                        &gBeta, &denseMatC, &control ) );
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

   hcsparseCsrMatrix gMatA;
   hcsparseCsrMatrix gMatB;
   hcsparseCsrMatrix gMatC;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view; 
   hcsparseControl control(accl_view);

   hcsparseSetup();
   hcsparseInitCsrMatrix(&gMatA);
   hcsparseInitCsrMatrix(&gMatB);
   hcsparseInitCsrMatrix(&gMatC);

   gMatA.offValues = 0;
   gMatA.offColInd = 0;
   gMatA.offRowOff = 0;

   gMatB.offValues = 0;
   gMatB.offValues = 0;
   gMatB.offColInd = 0;

   gMatC.offRowOff = 0;
   gMatC.offColInd = 0;
   gMatC.offRowOff = 0;

   array_view<float> av_values_A(nnzA, csrValA);
   array_view<int> av_rowOff_A(m+1, csrRowPtrA);
   array_view<int> av_colIndices_A(nnzA, csrColIndA);

   gMatA.values = &av_values_A;
   gMatA.rowOffsets = &av_rowOff_A;
   gMatA.colIndices = &av_colIndices_A;

   array_view<float> av_values_B(nnzB, csrValB);
   array_view<int> av_rowOff_B(k+1, csrRowPtrB);
   array_view<int> av_colIndices_B(nnzB, csrColIndB);

   gMatB.values = &av_values_B;
   gMatB.rowOffsets = &av_rowOff_B;
   gMatB.colIndices = &av_colIndices_B;

   int nnz;
   if (nnzA>nnzB)
   {
      nnz = nnzA;
   }
   else
   {
      nnz = nnzB;
   }

   array_view<float> av_values_C(nnz, csrValC);
   array_view<int> av_rowOff_C(m+1, csrRowPtrC);
   array_view<int> av_colIndices_C(nnz, csrColIndC);

   gMatC.values = &av_values_C;
   gMatC.rowOffsets = &av_rowOff_C;
   gMatC.colIndices = &av_colIndices_C;

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsrSpGemm(&gMatA, 
                                                           &gMatB, 
                                                           &gMatC,
                                                           &control ));
}


//Matrix conversion routines

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const float           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      float           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   hcdenseMatrix gMat;
   hcsparseCsrMatrix gCsrMat;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view; 
   hcsparseControl control(accl_view);

   hcsparseSetup();

   hcdenseInitMatrix(&gMat);
   gMat.offValues = 0;
   array_view<float> av_A_values(m*n, A);
   gMat.values = &av_A_values;
   gMat.num_rows = m;
   gMat.num_cols = n;

   hcsparseInitCsrMatrix(&gCsrMat);
   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   int num_nonzero =0;

   for (int i = 0; i < m; ++i)
   {
      num_nonzero += nnzPerRow[i];
   }

   array_view<float> av_csr_values(num_nonzero, csrValA);
   array_view<int> av_csr_rowOff(m+1, csrRowPtrA);
   array_view<int> av_csr_colIndices(num_nonzero, csrColIndA);

   gCsrMat.values = &av_csr_values;
   gCsrMat.rowOffsets = &av_csr_rowOff;
   gCsrMat.colIndices = &av_csr_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseSdense2csr( &gMat, 
                                                           &gCsrMat,
                                                           &control ) ); 
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, 
                                                      const hipsparseMatDescr_t descrA, 
                                                      const double           *A, 
                                                      int lda, const int *nnzPerRow, 
                                                      double           *csrValA, 
                                                      int *csrRowPtrA, int *csrColIndA) {

   hcdenseMatrix gMat;
   hcsparseCsrMatrix gCsrMat;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view; 
   hcsparseControl control(accl_view);

   hcsparseSetup();

   hcdenseInitMatrix(&gMat);
   gMat.offValues = 0;
   array_view<double> av_A_values(m*n, A);
   gMat.values = &av_A_values;
   gMat.num_rows = m;
   gMat.num_cols = n;

   hcsparseInitCsrMatrix(&gCsrMat);
   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;

   int num_nonzero =0;

   for (int i = 0; i < m; ++i)
   {
      num_nonzero += nnzPerRow[i];
   }

   array_view<double> av_csr_values(num_nonzero, csrValA);
   array_view<int> av_csr_rowOff(m+1, csrRowPtrA);
   array_view<int> av_csr_colIndices(num_nonzero, csrColIndA);

   gCsrMat.values = &av_csr_values;
   gCsrMat.rowOffsets = &av_csr_rowOff;
   gCsrMat.colIndices = &av_csr_colIndices;


   return hipHCSPARSEStatusToHIPStatus(hcsparseDdense2csr( &gMat, 
                                                           &gCsrMat,
                                                           &control ) ); 
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const float             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    float *A, int lda){

   hcsparseCsrMatrix gCsrMat;
   hcdenseMatrix gMat;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view; 
   hcsparseControl control(accl_view);

   hcsparseSetup();

   hcsparseInitCsrMatrix(&gCsrMat);
   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;
   
   hcdenseInitMatrix(&gMat);
   gMat.offValues = 0;
   array_view<float> av_A_values(m*n, A);
   gMat.values = &av_A_values;
   gMat.num_rows = m;
   gMat.num_cols = n;

   int num_nonzero = csrRowPtrA[m] - csrRowPtrA[0];

   array_view<float> av_csr_values(num_nonzero, csrValA);
   array_view<int> av_csr_rowOff(m+1, csrRowPtrA);
   array_view<int> av_csr_colIndices(num_nonzero, csrColIndA);

   gCsrMat.values = &av_csr_values;
   gCsrMat.rowOffsets = &av_csr_rowOff;
   gCsrMat.colIndices = &av_csr_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseScsr2dense( &gCsrMat, 
                                                            &gMat, 
                                                            &control ));
}

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, 
                                                    const hipsparseMatDescr_t descrA, 
                                                    const double             *csrValA, 
                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                    double *A, int lda){

   hcsparseCsrMatrix gCsrMat;
   hcdenseMatrix gMat;

   accelerator acc = accelerator(accelerator::default_accelerator);
   accelerator_view accl_view = acc.default_view; 
   hcsparseControl control(accl_view);

   hcsparseSetup();

   hcsparseInitCsrMatrix(&gCsrMat);
   gCsrMat.offValues = 0;
   gCsrMat.offColInd = 0;
   gCsrMat.offRowOff = 0;
   
   hcdenseInitMatrix(&gMat);
   gMat.offValues = 0;
   array_view<double> av_A_values(m*n, A);
   gMat.values = &av_A_values;
   gMat.num_rows = m;
   gMat.num_cols = n;

   int num_nonzero = csrRowPtrA[m] - csrRowPtrA[0];

   array_view<double> av_csr_values(num_nonzero, csrValA);
   array_view<int> av_csr_rowOff(m+1, csrRowPtrA);
   array_view<int> av_csr_colIndices(num_nonzero, csrColIndA);

   gCsrMat.values = &av_csr_values;
   gCsrMat.rowOffsets = &av_csr_rowOff;
   gCsrMat.colIndices = &av_csr_colIndices;

   return hipHCSPARSEStatusToHIPStatus(hcsparseDcsr2dense( &gCsrMat, 
                                                            &gMat, 
                                                            &control ));
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int *cooRowIndA, 
                                                  int nnz, int m, int *csrRowPtrA, 
                                                  hipsparseIndexBase_t idxBase){

   return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int *csrRowPtrA,
                                                  int nnz, int m, int *cooRowIndA, 
                                                  hipsparseIndexBase_t idxBase){

  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif

