#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include <cmath>
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csr_dense_conv_double_test, func_check)
{
    const char* filename = "./../../../../../test/gtest/src/input.mtx";
    int num_nonzero, num_row, num_col;
    double *values = NULL;
    int *rowOffsets = NULL;
    int *colIndices = NULL;

    if (!(hcsparseCsrMatrixfromFile<double>(filename, false, &values, &rowOffsets, &colIndices,
                                            &num_row, &num_col, &num_nonzero))) {
      std::cout << "Error reading the matrix file" << std::endl;
      exit(1);
    }

    std::cout << "num_row : " << num_row << "values " << values[5] << std::endl;

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

    double *csrValA = NULL;
    int *csrRowPtrA = NULL;
    int *csrColIndA = NULL;
    double *A = NULL;
    hipError_t err;

    err = hipMalloc(&csrValA, num_nonzero * sizeof(double));
    err = hipMalloc(&csrRowPtrA, (num_row+1) * sizeof(int));
    err = hipMalloc(&csrColIndA, num_nonzero * sizeof(int));
    err = hipMalloc(&A, num_row*num_col * sizeof(double));

    double *csr_val = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_colInd = (int*)calloc(num_nonzero, sizeof(int));    

    double *csr_res_val = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_res_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_colInd = (int*)calloc(num_nonzero, sizeof(int));    

    hipMemcpy(csrValA, values, num_nonzero * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(csrRowPtrA, rowOffsets, (num_row+1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(csrColIndA, colIndices, num_nonzero * sizeof(int), hipMemcpyHostToDevice);

    status1 = hipsparseDcsr2dense(handle, num_row, num_col,
                                 descrA, csrValA, csrRowPtrA, csrColIndA, A, num_row);
    
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2dense conversion "<<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csr_val, csrValA, num_nonzero * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(csr_rowPtr, csrRowPtrA, (num_row+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csr_colInd, csrColIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    int nnzperrow = 0;
    status1 = hipsparseDdense2csr(handle, num_row, num_col,
                                 descrA, A, num_row, &nnzperrow, csrValA, csrRowPtrA, csrColIndA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csr conversion "<<std::endl;
      exit(1);
    }
    hipDeviceSynchronize();

    hipMemcpy(csr_res_val, csrValA, num_nonzero * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(csr_res_rowPtr, csrRowPtrA, (num_row+1) * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(csr_res_colInd, csrColIndA, num_nonzero * sizeof(int), hipMemcpyDeviceToHost);

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        double diff = std::fabs(csr_val[i] - csr_res_val[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        double diff = std::fabs(csr_colInd[i] - csr_res_colInd[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_row+1; i++)
    {
        double diff = std::fabs(csr_rowPtr[i] - csr_res_rowPtr[i]);
        EXPECT_LT(diff, 0.01);
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

}
