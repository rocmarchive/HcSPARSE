#include <iostream>
#include <cmath>
#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csrmm_float_test, func_check)
{

    const char* filename = "./../../test/gtest/src/input.mtx";
    int num_nonzero, num_row_A, num_col_A;
    float *values = NULL;
    int *rowOffsets = NULL;
    int *colIndices = NULL;

     if ((hcsparseCsrMatrixfromFile<float>(filename, false, &values, &rowOffsets, &colIndices,
                                            &num_row_A, &num_col_A, &num_nonzero))) {
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

    int num_row_X, num_col_X, num_row_Y, num_col_Y;

    num_row_X = num_col_A;
    num_col_X = rand()%100;
    num_row_Y = num_row_A;
    num_col_Y = num_col_X;

    float *host_res = (float*) calloc(num_row_Y * num_col_Y, sizeof(float));
    float *host_X = (float*) calloc(num_row_X * num_col_X, sizeof(float));
    float *host_Y = (float*) calloc(num_row_Y * num_col_Y, sizeof(float));
    float *host_alpha = (float*) calloc(1, sizeof(float));
    float *host_beta = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_col_X * num_row_X; i++)
    {
       host_X[i] = rand()%100;
    }

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        host_res[i] = host_Y[i] = rand()%100;
    }

    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    float *gX;
    float *gY;
    float *gAlpha;
    float *gBeta;
    float *valA = NULL;
    int  *rowPtrA = NULL;
    int *colIndA = NULL;
    hipError_t err;

    err = hipMalloc(&gX, sizeof(float) * num_col_X * num_row_X);
    err = hipMalloc(&gY, sizeof(float) * num_row_Y * num_col_Y);
    err = hipMalloc(&gAlpha, sizeof(float) * 1);
    err = hipMalloc(&gBeta, sizeof(float) * 1);
    err = hipMalloc(&valA, sizeof(float) * num_nonzero);
    err = hipMalloc(&rowPtrA, sizeof(int) * (num_row_A+1));
    err = hipMalloc(&colIndA, sizeof(int) * num_nonzero);

    hipMemcpy(gX, host_X, sizeof(float) * num_col_X * num_row_X, hipMemcpyHostToDevice);
    hipMemcpy(gY, host_Y, sizeof(float) * num_row_Y * num_col_Y, hipMemcpyHostToDevice);
    hipMemcpy(gAlpha, host_alpha, sizeof(float) * 1, hipMemcpyHostToDevice);
    hipMemcpy(gBeta, host_beta, sizeof(float) * 1, hipMemcpyHostToDevice);
    hipMemcpy(valA, values, sizeof(float) * num_nonzero, hipMemcpyHostToDevice);
    hipMemcpy(rowPtrA, rowOffsets, sizeof(int) * (num_row_A+1), hipMemcpyHostToDevice);
    hipMemcpy(colIndA, colIndices, sizeof(int) * num_nonzero, hipMemcpyHostToDevice);

    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    int nnz = 0;

    status1 = hipsparseScsrmm(handle, transA, num_row_A, num_col_Y,
                            num_col_A, nnz, static_cast<const float*>(gAlpha), descrA,
                            valA, rowPtrA, colIndA, gX, num_col_X, 
                            static_cast<const float*>(gBeta), gY, num_col_Y);
    hipDeviceSynchronize();

    for (int col = 0; col < num_col_X; col++)
    {
        int indx = 0;
        for (int row = 0; row < num_row_A; row++)
        {
            float sum = 0.0;
            for (; indx < rowOffsets[row+1]; indx++)
            {
                sum += host_alpha[0] * host_X[colIndices[indx] * num_col_X + col] * values[indx];
            }
            host_res[row * num_col_Y + col] = sum + host_beta[0] * host_res[row * num_col_Y + col];
        }
    }

    hipMemcpy(host_Y, gY, sizeof(float) * num_row_Y * num_col_Y, hipMemcpyDeviceToHost);

    bool isPassed = 1;

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        float diff = std::abs(host_res[i] - host_Y[i]);
//        std::cout << i << ": " << "h = " << host_res[i] << " d: " << host_Y[i] << std::endl;
//        EXPECT_LT(diff, 0.01);
    }

    status1 = hipsparseDestroyMatDescr(descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      exit(1);
    }

    status1 = hipsparseDestroy(handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      exit(1);
    }
   
    /* End - Test of New APIs */

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
    hipFree(gAlpha);
    hipFree(gBeta);
    hipFree(valA);
    hipFree(rowPtrA);
    hipFree(colIndA);

}
