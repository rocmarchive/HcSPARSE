#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include "hc_am.hpp"
#include "gtest/gtest.h"

TEST(csr_dense_conv_double_test, func_check)
{
    hcsparseCsrMatrix gCsrMat;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    const char* filename = "./../../../../../test/gtest/src/input.mtx";
    int num_nonzero, num_row, num_col;
    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }

    gCsrMat.values = (double*) am_alloc(num_nonzero * sizeof(double), acc[1], 0);
    gCsrMat.rowOffsets = (int*) am_alloc((num_row+1) * sizeof(int), acc[1], 0);
    gCsrMat.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    hcsparseDCsrMatrixfromFile(&gCsrMat, filename, &control, false);

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

    double* csrValA = (double*) am_alloc(num_nonzero * sizeof(double), handle->currentAccl, 0);
    int* csrRowPtrA = (int*) am_alloc((num_row+1) * sizeof(int), handle->currentAccl, 0);
    int* csrColIndA = (int*) am_alloc(num_nonzero * sizeof(int), handle->currentAccl, 0);
    double* A = (double*) am_alloc(num_row*num_col * sizeof(double), handle->currentAccl, 0);

    double *csr_val = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_colInd = (int*)calloc(num_nonzero, sizeof(int));    

    double *csr_res_val = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_res_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_colInd = (int*)calloc(num_nonzero, sizeof(int));    

    control.accl_view.copy(gCsrMat.values, csrValA, num_nonzero * sizeof(double));
    control.accl_view.copy(gCsrMat.rowOffsets, csrRowPtrA, (num_row+1) * sizeof(int));
    control.accl_view.copy(gCsrMat.colIndices, csrColIndA, num_nonzero * sizeof(int));

    status1 = hipsparseDcsr2dense(handle, num_row, num_col,
                                 descrA, csrValA, csrRowPtrA, csrColIndA, A, num_row);
    control.accl_view.wait();
    
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2dense conversion "<<std::endl;
      exit(1);
    }

    control.accl_view.copy(csrValA, csr_val, num_nonzero * sizeof(double));
    control.accl_view.copy(csrRowPtrA, csr_rowPtr, (num_row+1) * sizeof(int));
    control.accl_view.copy(csrColIndA, csr_colInd, num_nonzero * sizeof(int));

    int nnzperrow = 0;
    status1 = hipsparseDdense2csr(handle, num_row, num_col,
                                 descrA, A, num_row, &nnzperrow, csrValA, csrRowPtrA, csrColIndA);
    control.accl_view.wait();
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csr conversion "<<std::endl;
      exit(1);
    }

    control.accl_view.copy(csrValA, csr_res_val, num_nonzero * sizeof(double));
    control.accl_view.copy(csrRowPtrA, csr_res_rowPtr, (num_row+1) * sizeof(int));
    control.accl_view.copy(csrColIndA, csr_res_colInd, num_nonzero * sizeof(int));

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        double diff = std::abs(csr_val[i] - csr_res_val[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        double diff = std::abs(csr_colInd[i] - csr_res_colInd[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_row+1; i++)
    {
        double diff = std::abs(csr_rowPtr[i] - csr_res_rowPtr[i]);
        EXPECT_LT(diff, 0.01);
    }

    free(csr_val);
    free(csr_rowPtr);
    free(csr_colInd);
    free(csr_res_val);
    free(csr_res_rowPtr);
    free(csr_res_colInd);
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);

}
