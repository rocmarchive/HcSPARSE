#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include "gtest/gtest.h"

TEST(geamNnz_test, func_check)
{
     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;
    hipsparseMatDescr_t descrA;
    hipsparseMatDescr_t descrB;
    hipsparseMatDescr_t descrC;

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }

    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;

    int m = 64;
    int n = 64;

    status1 = hipsparseCreateMatDescr(&descrA);
    status1 = hipsparseCreateMatDescr(&descrB);
    status1 = hipsparseCreateMatDescr(&descrC);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      exit(1);
    }

    float *devA = NULL;
    float *devB = NULL; 
    int *nnzPerRowColumnA = NULL;
    int *nnzA = NULL;
    int *nnzPerRowColumnB = NULL;
    int *nnzB = NULL;
    hipError_t err;

    err = hipMalloc(&devA, sizeof(float)*m*n);
    err = hipMalloc(&devB, sizeof(float)*m*n);
    err = hipMalloc(&nnzPerRowColumnA, sizeof(int)*m);
    err = hipMalloc(&nnzA, sizeof(int) * 1);
    err = hipMalloc(&nnzPerRowColumnB, sizeof(int)*m);
    err = hipMalloc(&nnzB, sizeof(int) * 1);

    float *hostA = (float*)calloc (m*n, sizeof(float));
    float *hostB = (float*)calloc (m*n, sizeof(float));
    float *hostC = (float*)calloc (m*n, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < m*n; i++)
    {
        hostA[i] = rand()%2;
    }    

    srand (time(NULL));
    for (int i = 0; i < m*n; i++)
    {
        hostB[i] = rand()%2;
    }
    
    hipMemcpy(devA, hostA, m*n*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(devB, hostB, m*n*sizeof(float), hipMemcpyHostToDevice);

    hipsparseStatus_t stat = hipsparseSnnz(handle, dir, m, n, descrA, devA, m,
                                         nnzPerRowColumnA, nnzA);
    stat = hipsparseSnnz(handle, dir, m, n, descrB, devB, m,
                         nnzPerRowColumnB, nnzB);
    hipDeviceSynchronize();

    int nnzA_h, nnzB_h;
    hipMemcpy(&nnzA_h, nnzA, 1*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&nnzB_h, nnzB, 1*sizeof(int), hipMemcpyDeviceToHost);

    float *csrValA = NULL;
    int *csrRowPtrA = NULL;
    int *csrColIndA = NULL;
    float *csrValB = NULL;
    int *csrRowPtrB = NULL;
    int *csrColIndB = NULL;
    int *csrRowPtrC = NULL;
    int *nnzC = NULL;

    err = hipMalloc(&csrValA, sizeof(float)*nnzA_h);
    err = hipMalloc(&csrRowPtrA, sizeof(int)*(m+1));
    err = hipMalloc(&csrColIndA, sizeof(int)*nnzA_h);
    err = hipMalloc(&csrValB, sizeof(float)*nnzB_h);
    err = hipMalloc(&csrRowPtrB, sizeof(int)*(m+1));
    err = hipMalloc(&csrColIndB, sizeof(int)*nnzB_h);
    err = hipMalloc(&csrRowPtrC, sizeof(int)*(m+1));
    err = hipMalloc(&nnzC, sizeof(int)*1);

    stat = hipsparseSdense2csr(handle, m, n, descrA, devA, m, nnzPerRowColumnA,
                                                 csrValA, csrRowPtrA, csrColIndA);

    stat = hipsparseSdense2csr(handle, m, n, descrB, devB, m, nnzPerRowColumnB,
                                                 csrValB, csrRowPtrB, csrColIndB);

    stat = hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA_h, csrRowPtrA, csrColIndA,
                                descrB, nnzB_h, csrRowPtrB, csrColIndB,
                                descrC, csrRowPtrC, nnzC);

    hipDeviceSynchronize();

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            hostC[i*n + j] += hostA[i*n + j] * hostB[i*n + j];
        }
    }

    float *devC = NULL;
    int *nnzPerRowColumnC_h = NULL;
    int *nnzC_h = NULL;
    float *csrValC_h = NULL;
    int *csrRowPtrC_h = NULL;
    int *csrColIndC_h = NULL;

    hipMalloc(&devC, sizeof(float)*m*n);
    hipMemcpy(devC, hostC, m*n*sizeof(float), hipMemcpyHostToDevice);

    err = hipMalloc(&nnzPerRowColumnC_h, sizeof(int)*m);
    err = hipMalloc(&nnzC_h, sizeof(int) * 1);

    stat = hipsparseSnnz(handle, dir, m, n, descrC, devC, m,
                         nnzPerRowColumnC_h, nnzC_h);

    int nnzC_ref_host;
    hipMemcpy(&nnzC_ref_host, nnzC_h, 1*sizeof(int), hipMemcpyDeviceToHost);

    err = hipMalloc(&csrValC_h, sizeof(float)*nnzC_ref_host);
    err = hipMalloc(&csrRowPtrC_h, sizeof(int)*(m+1));
    err = hipMalloc(&csrColIndC_h, sizeof(int)*nnzC_ref_host);

    stat = hipsparseSdense2csr(handle, m, n, descrC, devC, m, nnzPerRowColumnC_h,
                                                 csrValC_h, csrRowPtrC_h, csrColIndC_h);

    int *csrRowPtrC_host = (int*) malloc ((m+1)*sizeof(int));
    int *csrRowPtrC_ref_host = (int*) malloc ((m+1)*sizeof(int));
    int nnzC_host;

    hipMemcpy(csrRowPtrC_host, csrRowPtrC, (m+1)*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csrRowPtrC_ref_host, csrRowPtrC_h, (m+1)*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&nnzC_host, nnzC, 1*sizeof(int), hipMemcpyDeviceToHost);

    bool ispassed = 1;
    for (int i = 0; i < m+1; i++) {
      EXPECT_EQ(csrRowPtrC_ref_host[i], csrRowPtrC_host[i]);
    }
    
    EXPECT_EQ(nnzC_ref_host, nnzC_host);
 
    status1 = hipsparseDestroyMatDescr(descrA);
    status1 = hipsparseDestroyMatDescr(descrB);
    status1 = hipsparseDestroyMatDescr(descrC);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      exit(1);
    }

    status1 = hipsparseDestroy(handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
      exit(1);
    }
   
    hipFree(devA);
    hipFree(devB);
    hipFree(devC);
    hipFree(nnzPerRowColumnA);
    hipFree(nnzA);
    hipFree(nnzPerRowColumnB);
    hipFree(nnzB);
    hipFree(csrValA);
    hipFree(csrRowPtrA);
    hipFree(csrColIndA);
    hipFree(csrValB);
    hipFree(csrRowPtrB);
    hipFree(csrColIndB);
    hipFree(csrRowPtrC);
    hipFree(nnzC);
    hipFree(nnzPerRowColumnC_h);
    hipFree(nnzC_h);
    hipFree(csrValC_h);
    hipFree(csrRowPtrC_h);
    hipFree(csrColIndC_h);
    free(hostA);
    free(hostB);
    free(hostC);
    free(csrRowPtrC_host);
    free(csrRowPtrC_ref_host);
}
