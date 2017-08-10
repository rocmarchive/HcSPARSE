#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include "gtest/gtest.h"

TEST(nnz_float_test, func_check)
{
     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;
    hipsparseMatDescr_t descrA;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }

    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;

    int m = 64;
    int n = 259;
    int lda = m;

    status1 = hipsparseCreateMatDescr(&descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      exit(1);
    }

    float *devA = NULL;
    int *nnzPerRowColumn = NULL;
    int *nnz = NULL;
    hipError_t err;

    err = hipMalloc(&devA, sizeof(float)*m*n);
    err = hipMalloc(&nnzPerRowColumn, sizeof(float)*m);
    err = hipMalloc(&nnz, sizeof(int) * 1);

    float *hostA = (float*)calloc (m*n, sizeof(float));
    int *nnzPerRowColumn_h = (int *)calloc(m, sizeof(int));
    int *nnzPerRowColumn_res = (int *)calloc(m, sizeof(int));
    int nnz_res, nnz_h;

    srand (time(NULL));
    for (int i = 0; i < m*n; i++)
    {
        hostA[i] = rand()%100;
    }    

    hipMemcpy(devA, hostA, m*n*sizeof(float));

    hipsparseStatus_t stat = hipsparseDnnz(handle, dir, m, n, descrA, devA, lda,
                                         nnzPerRowColumn, nnz);
    hipDeviceSynchronize();

    hipMemcpy(nnzPerRowColumn_res, nnzPerRowColumn, m*sizeof(int));
    hipMemcpy(&nnz_res, &nnz, 1*sizeof(int));

    for (int i = 0; i < m; i++) {
      int rowCount = 0;
      for (int j = 0; j < n; j++) {
         if ( hostA[i*n+j] != 0) {
           rowCount++;
           nnz_h++;
         }
      }
      nnzPerRowColumn_h[i] = rowCount;
    }

    bool ispassed = 1;
    for (int i = 0; i < m; i++) {
      float diff = std::abs(nnzPerRowColumn_h[i] - nnzPerRowColumn_res[i]);
      EXPECT_LT(diff, 0.01);
    }
    
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
