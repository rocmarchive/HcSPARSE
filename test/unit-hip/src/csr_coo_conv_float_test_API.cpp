#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "hcsparse.h"
#include <iostream>
#include "hc_am.hpp"
#include "gtest/gtest.h"

TEST(csr_coo_conv_float_test, func_check)
{

     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();
    hcsparseControl control(av);

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

    std::vector<int> rowbuf = {  0,  2,  4,  8,  9, 11, 14, 16, 18, 19, 21, 24, 26, 28, 30, 32, 33};
  
    std::vector<int> col = { 5, 11,  5,  6,  0,  2,  5, 10, 14, 12, 14,  8, 12, 13,  8, 12,  3,
                             4,  0,  8, 12,  0,  7, 11,  2,  5,  8, 15,  5, 13,  0, 14,  6};
  
    std::vector<float> data = {-0.30078344,  0.87754357, -1.26799387, -0.35865804,  0.23114593,
                                1.22862246,  0.87208668,  0.45350665,  1.27449053, -1.384495  ,
                                0.59667623,  1.15120472,  0.62385128,  0.11424487,  1.24279024,
                                1.92432527, -0.71391821,  1.11341969,  0.75884568, -0.09533611,
                                1.30239323, -1.55654387, -1.68282037,  0.42669599,  1.67691593,
                               -0.36330267, -1.12321765,  0.32965108,  1.04276605, -2.22129573,
                                0.66819518, -0.35709961,  0.53445514};

    std::vector<int> row_ref = { 0,  0,  1,  1,  2,  2,  2,  2,  3,  4,  4,  5,  5,  5,  6,  6,
                                 7,  7,  8,  9,  9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15};

    int num_row = 16;
    int num_col = 16;
    int num_nonzero = col.size();
    int* csrRowPtrA = (int*) am_alloc((num_row+1) * sizeof(int), handle->currentAccl, 0);
    int* cooRowIndA = (int*) am_alloc((num_nonzero) * sizeof(int), handle->currentAccl, 0);

    handle->currentAcclView.copy(rowbuf.data(), csrRowPtrA, (num_row + 1) * sizeof(int));

    int *csr_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *coo_rowInd = (int*)calloc(num_nonzero, sizeof(int));

    status1 = hipsparseXcsr2coo(handle, csrRowPtrA, num_nonzero, num_row,
                               cooRowIndA, HCSPARSE_INDEX_BASE_ZERO);

    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2coo conversion "<<std::endl;
      exit(1);
    }

    control.accl_view.copy(csrRowPtrA, csr_rowPtr, (num_row+1) * sizeof(int));
    control.accl_view.copy(cooRowIndA, coo_rowInd, (num_nonzero) * sizeof(int));

    status1 = hipsparseXcoo2csr(handle, cooRowIndA, num_nonzero, num_row,
                               csrRowPtrA, HCSPARSE_INDEX_BASE_ZERO);

    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error coo2csr conversion "<<std::endl;
      exit(1);
    }

    control.accl_view.copy(csrRowPtrA, csr_res_rowPtr, (num_row+1) * sizeof(int));

    for (int i = 0; i < num_row+1; i++)
    {
        int diff = std::abs(csr_rowPtr[i] - csr_res_rowPtr[i]);
        EXPECT_LT(diff, 0.01);
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        int diff = std::abs(coo_rowInd[i] - row_ref[i]);
        EXPECT_LT(diff, 0.01);
    }

    hipsparseDestroyMatDescr(descrA);
    hipsparseDestroy(handle);
  
    free(csr_rowPtr);
    free(csr_res_rowPtr);
    am_free(csrRowPtrA);
    am_free(cooRowIndA);
}
