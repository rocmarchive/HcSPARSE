#include <iostream>
#include <cmath>
#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csrmv_float_test, func_check)
{
#if 0
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
#else

   int num_nonzero =9;
   int num_row = 4;
   int num_col = 5;

    float* values = (float*)calloc(num_nonzero, sizeof(float));
    int* rowOffsets = (int*)calloc(num_row+1, sizeof(int));
    int* colIndices = (int*)calloc(num_nonzero, sizeof(int));

    values[0] = 1;
    values[1] = 4;
    values[2] = 2;
    values[3] = 3;
    values[4] = 5;
    values[5] = 7;
    values[6] = 8;
    values[7] = 9;
    values[8] = 6;

    rowOffsets[0] = 0;
    rowOffsets[1] = 2;
    rowOffsets[2] = 4;
    rowOffsets[3] = 7;
    rowOffsets[4] = 9;

    colIndices[0] = 0;
    colIndices[1] = 1;
    colIndices[2] = 1;
    colIndices[3] = 2;
    colIndices[4] = 0;
    colIndices[5] = 3;
    colIndices[6] = 4;
    colIndices[7] = 2;
    colIndices[8] = 4;

#endif

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
       host_X[i] = 1; //rand()%100;
    } 

    for (int i = 0; i < num_row; i++)
    {
        host_res[i] = host_Y[i] = 1; //rand()%100;
    }

    host_alpha[0] = 1; //rand()%100;
    host_beta[0] = 1; //rand()%100;

    float *gX;
    float *gY;
    float *valA = NULL;
    int  *rowPtrA = NULL;
    int *colIndA = NULL;
    float *tvalA = NULL;
    int  *trowIndA = NULL;
    int *tcolPtrA = NULL;
    float *A = NULL;
    hipError_t err;

    err = hipMalloc(&gX, sizeof(float) * num_col);
    err = hipMalloc(&gY, sizeof(float) * num_row);
    err = hipMalloc(&valA, sizeof(float) * num_nonzero);
    err = hipMalloc(&rowPtrA, sizeof(int) * (num_row+1));
    err = hipMalloc(&colIndA, sizeof(int) * num_nonzero);
    err = hipMalloc(&tvalA, sizeof(float) * num_nonzero);
    err = hipMalloc(&trowIndA, sizeof(int) * (num_col+1));
    err = hipMalloc(&tcolPtrA, sizeof(int) * num_nonzero);
    err = hipMalloc(&A, sizeof(float) * (num_row*num_col));

    hipMemcpy(gX, host_X, sizeof(float) * num_col, hipMemcpyHostToDevice);
    hipMemcpy(valA, values, sizeof(float) * num_nonzero, hipMemcpyHostToDevice);
    hipMemcpy(rowPtrA, rowOffsets, sizeof(int) * (num_row+1), hipMemcpyHostToDevice);
    hipMemcpy(colIndA, colIndices, sizeof(int) * num_nonzero, hipMemcpyHostToDevice);

#if 0
    cusparseStatus_t stat = cusparseScsr2csc(handle, num_row, num_col, num_nonzero,
                                             valA, rowPtrA, colIndA, tvalA, trowIndA, tcolPtrA,
                                             CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    if (stat != CUSPARSE_STATUS_SUCCESS) {
             std::cout << "csr2csc error" << stat << std::endl;
             exit(1);
    }
#endif

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
        std::cout << "i : " << i << " y_h : " << host_res[i] << " y_d : " << host_Y[i] <<std::endl;
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
