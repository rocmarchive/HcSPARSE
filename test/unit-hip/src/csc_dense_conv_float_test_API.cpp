#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"

TEST(csc_dense_conv_float_test, func_check)
{

    int num_row = 4;
    int num_col = 5;
    int num_nonzero = 9;

    float* valPtr = (float*)calloc(num_row*num_col, sizeof(float));
    int* colPtr = (int*)calloc(num_col+1, sizeof(int));
    int* rowInd = (int*)calloc(num_nonzero, sizeof(int));

    valPtr[0] = 1;
    valPtr[1] = 5;
    valPtr[2] = 4;
    valPtr[3] = 2;
    valPtr[4] = 3;
    valPtr[5] = 9;
    valPtr[6] = 7;
    valPtr[7] = 8;
    valPtr[8] = 6;

    colPtr[0] = 0;
    colPtr[1] = 2;
    colPtr[2] = 4;
    colPtr[3] = 6;
    colPtr[4] = 7;
    colPtr[5] = 9;

    rowInd[0] = 0;
    rowInd[1] = 2;
    rowInd[2] = 0;
    rowInd[3] = 1;
    rowInd[4] = 1;
    rowInd[5] = 3;
    rowInd[6] = 2;
    rowInd[7] = 2;
    rowInd[8] = 3;

     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }

    hipsparseMatDescr_t descrA;

    status1 = hipsparseCreateMatDescr(&descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      exit(1);
    }

    float* cscValA = NULL;
    int *cscColPtrA = NULL;
    int *cscRowIndA = NULL;
    float *A = NULL;
    hipError_t err;

    err = hipMalloc(&cscValA, num_nonzero * sizeof(float));
    err = hipMalloc(&cscColPtrA, (num_col+1) * sizeof(int));
    err = hipMalloc(&cscRowIndA, num_nonzero * sizeof(int));
    err = hipMalloc(&A, num_row*num_col * sizeof(float));

    float *csc_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_colPtr = (int*)calloc(num_col+1, sizeof(int));
    int *csc_rowInd = (int*)calloc(num_nonzero, sizeof(int));    

    float *csc_res_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_res_colPtr = (int*)calloc(num_col+1, sizeof(int));
    int *csc_res_rowInd = (int*)calloc(num_nonzero, sizeof(int));    
     
    hipMemcpy(cscValA, valPtr, num_nonzero * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(cscColPtrA, colPtr, (num_col+1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(cscRowIndA, rowInd, num_nonzero * sizeof(int), hipMemcpyHostToDevice);

    status1 = hipsparseScsc2dense(handle, num_row, num_col,
                                 descrA, cscValA, cscColPtrA, cscRowIndA, A, num_col);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csc2dense conversion "<<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csc_val, cscValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csc_colPtr, cscColPtrA, (num_col+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csc_rowInd, cscRowIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    hipMemset(cscValA, 0, num_nonzero * sizeof(float));
    hipMemset(cscColPtrA, 0, (num_col+1) * sizeof(int));
    hipMemset(cscRowIndA, 0, num_nonzero * sizeof(int));

    int nnzperrow = 8644/900;
    status1 = hipsparseSdense2csc(handle, num_row, num_col,
                                 descrA, A, num_row, &nnzperrow, cscValA, cscColPtrA, cscRowIndA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csc conversion " << status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csc_res_val, cscValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_colPtr, cscColPtrA, (num_col+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_rowInd, cscRowIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::abs(csc_val[i] - csc_res_val[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::abs(csc_rowInd[i] - csc_res_rowInd[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_col+1; i++)
    {
        float diff = std::abs(csc_colPtr[i] - csc_res_colPtr[i]);
        EXPECT_LT(diff, 0.01);
    }

    free(csc_val);
    free(csc_colPtr);
    free(csc_rowInd);
    free(csc_res_val);
    free(csc_res_colPtr);
    free(csc_res_rowInd);
    hipFree(cscValA);
    hipFree(cscColPtrA);
    hipFree(cscRowIndA);

}
