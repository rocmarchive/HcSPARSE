#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include "gtest/gtest.h"

TEST(nnz_double_test, func_check)
{
     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;
    hipsparseMatDescr_t descrA;

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }

    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;

    int m = 64;
    int n = 64;
    int lda = n;

    status1 = hipsparseCreateMatDescr(&descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      exit(1);
    }

    double *devA = NULL;
    int *nnzPerRowColumn = NULL;
    int *nnz = NULL;
    hipError_t err;

    err = hipMalloc(&devA, sizeof(double)*m*n);
    err = hipMalloc(&nnzPerRowColumn, sizeof(double)*m);
    err = hipMalloc(&nnz, sizeof(int) * 1);

    double *hostA = (double*)calloc (m*n, sizeof(double));
    int *nnzPerRowColumn_h = (int *)calloc(m, sizeof(int));
    int *nnzPerRowColumn_res = (int *)calloc(m, sizeof(int));
    int nnz_res, nnz_h = 0;

    srand (time(NULL));
    for (int i = 0; i < m*n; i++)
    {
        hostA[i] = rand()%100;
    }    

#if 0
    for (int i = 0; i < m; i++) {
      std::cout << i << ": \t";
      for (int j = 0; j < n; j++) {
        std::cout << hostA[i*n+j] << "\t";
      }
      std::cout << std::endl;
    }
#endif

    std::cout << std::endl;
    hipMemcpy(devA, hostA, m*n*sizeof(double), hipMemcpyHostToDevice);

    hipsparseStatus_t stat = hipsparseDnnz(handle, dir, n, m, descrA, devA, lda,
                                         nnzPerRowColumn, nnz);
    hipDeviceSynchronize();

    hipMemcpy(nnzPerRowColumn_res, nnzPerRowColumn, m*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&nnz_res, nnz, 1*sizeof(int), hipMemcpyDeviceToHost);

    for (int i = 0;i < m; i++) {
      int rowCount = 0;
      for (int j = 0; j < n; j++) {
         if ( hostA[i * n + j] != 0) {
           rowCount++;
         }
      }
      nnzPerRowColumn_h[i] = rowCount;
      nnz_h += rowCount;
    }

    bool ispassed = 1;
    for (int i = 0; i < m; i++) {
      double diff = std::abs(nnzPerRowColumn_h[i] - nnzPerRowColumn_res[i]);
      EXPECT_LT(diff, 0.01);
    }

    double diff = std::abs(nnz_res - nnz_h);
    EXPECT_LT(diff, 0.01); 
 
    status1 = hipsparseDestroyMatDescr(descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      exit(1);
    }

    status1 = hipsparseDestroy(handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
      exit(1);
    }
   
    hipFree(devA);
    hipFree(nnzPerRowColumn);
    hipFree(nnz);
    free(hostA);
    free(nnzPerRowColumn_h);
    free(nnzPerRowColumn_res);

}
