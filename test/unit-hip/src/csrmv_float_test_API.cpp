#include <iostream>
#include <cmath>
#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csrmv_float_test, func_check)
{
    const char* filename = "./../../test/gtest/src/input.mtx";

    int num_nonzero, num_row, num_col;
    float *values = NULL;
    int *rowOffsets = NULL;
    int *colIndices = NULL;

    if ((hcsparseCsrMatrixfromFile<float>(filename, false, &values, &rowOffsets, &colIndices,
                                            &num_row, &num_col, &num_nonzero))) {
      std::cout << "Error reading the matrix file" << std::endl;
      exit(1);
    }

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

    float *host_res = (float*) calloc(num_row, sizeof(float));
    float *host_X = (float*) calloc(num_col, sizeof(float));
    float *host_Y = (float*) calloc(num_row, sizeof(float));
    float *host_alpha = (float*) calloc(1, sizeof(float));
    float *host_beta = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_col; i++)
    {
       host_X[i] = rand()%100;
    } 

    for (int i = 0; i < num_row; i++)
    {
        host_res[i] = host_Y[i] = rand()%100;
    }

    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    float *gX;
    float *gY;
    float *valA = NULL;
    int  *rowPtrA = NULL;
    int *colIndA = NULL;
    float *A = NULL;
    hipError_t err;

    err = hipMalloc(&gX, sizeof(float) * num_col);
    err = hipMalloc(&gY, sizeof(float) * num_row);
    err = hipMalloc(&valA, sizeof(float) * num_nonzero);
    err = hipMalloc(&rowPtrA, sizeof(int) * (num_row+1));
    err = hipMalloc(&colIndA, sizeof(int) * num_nonzero);
    err = hipMalloc(&A, sizeof(float) * (num_row*num_col));

    hipMemcpy(gX, host_X, sizeof(float) * num_col, hipMemcpyHostToDevice);
    hipMemcpy(valA, values, sizeof(float) * num_nonzero, hipMemcpyHostToDevice);
    hipMemcpy(rowPtrA, rowOffsets, sizeof(int) * (num_row+1), hipMemcpyHostToDevice);
    hipMemcpy(colIndA, colIndices, sizeof(int) * num_nonzero, hipMemcpyHostToDevice);

    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    status1 = hipsparseScsrmv(handle, transA, num_row, num_col,
                              num_nonzero, (const float *)host_alpha,
                              (const hipsparseMatDescr_t)descrA, 
                              (const float*)valA, (const int*)rowPtrA,
                              (const int*)colIndA, (const float*)gX,
                              (const float*)host_beta, gY);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
       std::cout << "Error in csrmv operation " << status1 << std::endl;
       exit(1);
    }
    hipDeviceSynchronize();
     
    int col = 0;
    for (int row = 0; row < num_row; row++)
    {
        host_res[row] *= host_beta[0];
        for (; col < rowOffsets[row+1]; col++)
        {
            host_res[row] = host_alpha[0] * host_X[colIndices[col]] * values[col] + host_res[row];
        }
    }

    hipMemcpy(host_Y, gY, sizeof(float) * num_row, hipMemcpyDeviceToHost);

    bool isPassed = 1;  
 
    for (int i = 0; i < num_row; i++)
    {
        float diff = std::abs(host_res[i] - host_Y[i]);
 //       EXPECT_LT(diff, 0.01);
    }

    free(host_res);
    free(host_X);
    free(host_Y);
    free(host_alpha);
    free(host_beta);
    free(values);
    free(rowOffsets);
    free(colIndices);
    hipFree(gX);
    hipFree(gY);
    hipFree(values);
    hipFree(rowOffsets);
    hipFree(colIndices);
    hipFree(A);
}
