#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include <cmath>
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csr_csc_float_test, func_check)
{
    const char* filename = "./../../../../../test/gtest/src/input.mtx";
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
    float *csrValA = NULL;
    int *csrRowPtrA = NULL;
    int *csrColIndA = NULL;
    float *cscValA = NULL;
    int *cscColPtrA = NULL;
    int *cscRowIndA = NULL;
    int *nnzPerCol = NULL;
    float *A = NULL;
    int *nnz = NULL;
    hipError_t err;

    err = hipMalloc(&csrValA, num_nonzero * sizeof(float));
    err = hipMalloc(&csrRowPtrA, (num_row+1) * sizeof(int));
    err = hipMalloc(&csrColIndA, num_nonzero * sizeof(int));
    err = hipMalloc(&cscValA, num_nonzero * sizeof(float));
    err = hipMalloc(&cscColPtrA, (num_col+1) * sizeof(int));
    err = hipMalloc(&cscRowIndA, num_nonzero * sizeof(int));
    err = hipMalloc(&nnzPerCol, num_col * sizeof(int));
    err = hipMalloc(&A, num_row * num_col * sizeof(int));
    err = hipMalloc(&nnz, 1 * sizeof(int));

    float *csc_res_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_res_colPtr = (int*)calloc(num_col+1, sizeof(int));
    int *csc_res_rowInd = (int*)calloc(num_nonzero, sizeof(int));    
    float *csc_res_val1 = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_res_colPtr1 = (int*)calloc(num_col+1, sizeof(int));
    int *csc_res_rowInd1 = (int*)calloc(num_nonzero, sizeof(int));    

    hipMemcpy(csrValA, values, num_nonzero * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(csrRowPtrA, rowOffsets, (num_row+1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(csrColIndA, colIndices, num_nonzero * sizeof(int), hipMemcpyHostToDevice);

    status1 = hipsparseScsr2dense(handle, num_row, num_col,
                                 descrA, csrValA, csrRowPtrA, csrColIndA, A, num_row);
    
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2dense conversion "<<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    status1 = hipsparseSnnz(handle, HIPSPARSE_DIRECTION_COLUMN, num_row, num_col, descrA, A, num_col, nnzPerCol, nnz);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error nnz "<< status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    status1 = hipsparseSdense2csc(handle, num_row, num_col,
                                 descrA, A, num_col, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csr conversion "<< status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csc_res_val, cscValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_colPtr, cscColPtrA, (num_col+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_rowInd, cscRowIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    status1 = hipsparseScsr2csc(handle,num_row, num_col, 0, csrValA, csrRowPtrA, csrColIndA, cscValA, cscRowIndA, cscColPtrA, HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO); 
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2csc conversion "<< status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csc_res_val1, cscValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_colPtr1, cscColPtrA, (num_col+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csc_res_rowInd1, cscRowIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::fabs(csc_res_val[i] - csc_res_val1[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::fabs(csc_res_rowInd[i] - csc_res_rowInd1[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_col+1; i++)
    {
        float diff = std::fabs(csc_res_colPtr[i] - csc_res_colPtr1[i]);
        EXPECT_LT(diff, 0.01);
    }

    free(values);
    free(rowOffsets);
    free(colIndices);
    free(csc_res_val);
    free(csc_res_colPtr);
    free(csc_res_rowInd);
    free(csc_res_val1);
    free(csc_res_colPtr1);
    free(csc_res_rowInd1);
    hipFree(csrValA);
    hipFree(csrRowPtrA);
    hipFree(csrColIndA);
    hipFree(cscValA);
    hipFree(cscColPtrA);
    hipFree(cscRowIndA);
    hipFree(A);
    hipFree(nnzPerCol);
    hipFree(nnz);
}
