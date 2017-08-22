#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include <cmath>
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csr_dense_conv_float_test, func_check)
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

    float *csrValA = NULL;
    int *csrRowPtrA = NULL;
    int *csrColIndA = NULL;
    int *nnzPerRow = NULL;
    float *A = NULL;
    int *nnz = NULL;
    hipError_t err;

    err = hipMalloc(&csrValA, num_nonzero * sizeof(float));
    err = hipMalloc(&csrRowPtrA, (num_row+1) * sizeof(int));
    err = hipMalloc(&csrColIndA, num_nonzero * sizeof(int));
    err = hipMalloc(&nnzPerRow, num_row * sizeof(int));
    err = hipMalloc(&A, num_row * num_col * sizeof(int));
    err = hipMalloc(&nnz, 1 * sizeof(int));

    float *csr_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_colInd = (int*)calloc(num_nonzero, sizeof(int));    

    float *csr_res_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_res_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_colInd = (int*)calloc(num_nonzero, sizeof(int));    

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

    hipMemcpy(csr_val, csrValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csr_rowPtr, csrRowPtrA, (num_row+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csr_colInd, csrColIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    status1 = hipsparseSnnz(handle, HIPSPARSE_DIRECTION_ROW, num_row, num_col, descrA, A, num_row, nnzPerRow, nnz);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error nnz "<< status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    status1 = hipsparseSdense2csr(handle, num_row, num_col,
                                 descrA, A, num_row, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csr conversion "<< status1 <<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csr_res_val, csrValA, num_nonzero * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(csr_res_rowPtr, csrRowPtrA, (num_row+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csr_res_colInd, csrColIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::fabs(csr_val[i] - csr_res_val[i]);
//        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::fabs(csr_colInd[i] - csr_res_colInd[i]);
//        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_row+1; i++)
    {
        float diff = std::fabs(csr_rowPtr[i] - csr_res_rowPtr[i]);
//        EXPECT_LT(diff, 0.01);
    }

    free(values);
    free(rowOffsets);
    free(colIndices);
    free(csr_val);
    free(csr_rowPtr);
    free(csr_colInd);
    free(csr_res_val);
    free(csr_res_rowPtr);
    free(csr_res_colInd);
    hipFree(csrValA);
    hipFree(csrRowPtrA);
    hipFree(csrColIndA);
    hipFree(A);
    hipFree(nnzPerRow);
    hipFree(nnz);

}
