#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>
#include "mmio_wrapper.h"
#include "gtest/gtest.h"

TEST(csrmm_float_test, func_check)
{
    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());

    hcsparseControl control(accl_view);

    const char* filename = "./../../../../test/gtest/src/input.mtx";
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
    hcsparseHandle_t handle;
    hcsparseStatus_t status1;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();

    status1 = hcsparseCreate(&handle, &av);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }

    hcsparseMatDescr_t descrA;

    status1 = hcsparseCreateMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
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
    float *rowPtrA = NULL;
    float *colIndA = NULL;

    gX = am_alloc(sizeof(float) * num_col_X * num_row_X, acc[1], 0);
    gY = am_alloc(sizeof(float) * num_row_Y * num_col_Y, acc[1], 0);
    gAlpha = am_alloc(sizeof(float) * 1, acc[1], 0);
    gBeta = am_alloc(sizeof(float) * 1, acc[1], 0);
    valA = am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    rowPtrA = am_alloc((num_row_A+1) * sizeof(int), acc[1], 0);
    colIndA = am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    control.accl_view.copy(host_X, gX, sizeof(float) * num_col_X * num_row_X);
    control.accl_view.copy(host_Y, gY, sizeof(float) * num_row_Y * num_col_Y);
    control.accl_view.copy(host_alpha, gAlpha, sizeof(float) * 1);
    control.accl_view.copy(host_beta, gBeta, sizeof(float) * 1);
    control.accl_view.copy(values, valA, sizeof(float) * num_nonzero);
    control.accl_view.copy(rowOffsets, rowPtrA, sizeof(int) * (num_row_A+1));
    control.accl_view.copy(colIndices, colIndA, sizeof(int) * num_nonzero);

    hcsparseOperation_t transA = HCSPARSE_OPERATION_NON_TRANSPOSE;

    status1 = hcsparseScsrmm(handle, transA, num_row_A, num_col_Y,
                            num_col_A, num_nonzero, static_cast<const float*>(gAlpha), descrA,
                            static_cast<const float*>(valA),
                            (int *) rowPtrA,
                            (int *)colIndA, (float*)gX, num_col_A,
                            static_cast<const float*>(gBeta), (float *)gY, num_row_A);

    for (int col = 0; col < num_col_X; col++)
    {
        int indx = 0;
        for (int row = 0; row < num_row_A; row++)
        {
            float sum = 0.0;
            for (; indx < rowOffsets[row+1]; indx++)
            {
                sum += host_alpha[0] * host_X[colIndices[indx] + num_row_X * col] * values[indx];
            }
            host_res[row + num_row_A * col] = sum + host_beta[0] * host_res[row + num_row_A * col];
        }
    }

    control.accl_view.copy(gY, host_Y, sizeof(float) * num_row_Y * num_col_Y);

    bool isPassed = 1;

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        float diff = std::abs(host_res[i] - host_Y[i]);
        EXPECT_LT(diff, 0.01);
    }

    std::cout << (isPassed?"TEST PASSED":"TEST FAILED") << std::endl;

    status1 = hcsparseDestroyMatDescr(descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error destroy mat descrptr"<<std::endl;
      exit(1);
    }

    status1 = hcsparseDestroy(&handle);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
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
    am_free(gX);
    am_free(gY);
    am_free(gAlpha);
    am_free(gBeta);
    am_free(valA);
    am_free(rowPtrA);
    am_free(colIndA);
}
