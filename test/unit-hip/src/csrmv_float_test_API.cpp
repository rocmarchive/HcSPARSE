#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include "hcsparse.h"
#include <iostream>
#include "hc_am.hpp"
#include "gtest/gtest.h"

TEST(csrmv_float_test, func_check)
{
    hcsparseCsrMatrix gCsrMat;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    const char* filename = "./../../../../../test/gtest/src/input.mtx";

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(-1);
    } 

     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      exit(1);
    }
    std::cout << "Successfully initialized sparse library"<<std::endl;

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

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);

    float *gX = am_alloc(sizeof(float) * num_col, acc[1], 0);
    float *gY = am_alloc(sizeof(float) * num_row, acc[1], 0);
    float *gAlpha = am_alloc(sizeof(float) * 1, acc[1], 0);
    float *gBeta = am_alloc(sizeof(float) * 1, acc[1], 0);

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

    control.accl_view.copy(host_X, gX, sizeof(float) * num_col);
    control.accl_view.copy(host_Y, gY, sizeof(float) * num_row);
    control.accl_view.copy(host_alpha, gAlpha, sizeof(float) * 1);
    control.accl_view.copy(host_beta, gBeta, sizeof(float) * 1);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    float *values = (float*)calloc(num_nonzero, sizeof(float));
    int *rowOffsets = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCsrMat.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gCsrMat.rowOffsets = am_alloc(sizeof(int) * (num_row+1), acc[1], 0);
    gCsrMat.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);
   
    control.accl_view.copy(gCsrMat.values, values, sizeof(float) * num_nonzero);
    control.accl_view.copy(gCsrMat.rowOffsets, rowOffsets, sizeof(int) * (num_row+1));
    control.accl_view.copy(gCsrMat.colIndices, colIndices, sizeof(int) * num_nonzero);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }

    hipsparseOperation_t transA = HCSPARSE_OPERATION_NON_TRANSPOSE;
    int nnz = 0;

    status1 = hipsparseScsrmv(handle, transA, num_row, num_col,
                              nnz, static_cast<const float*>(gAlpha), descrA,
                              static_cast<const float*>(gCsrMat.values),
                              (int *) gCsrMat.rowOffsets,
                              (int *)gCsrMat.colIndices, (float*)gX,
                              static_cast<const float*>(gBeta), (float *)gY);

     
    int col = 0;
    for (int row = 0; row < num_row; row++)
    {
        host_res[row] *= host_beta[0];
        for (; col < rowOffsets[row+1]; col++)
        {
            host_res[row] = host_alpha[0] * host_X[colIndices[col]] * values[col] + host_res[row];
        }
    }

    control.accl_view.copy(gY, host_Y, sizeof(float) * num_row);

    bool isPassed = 1;  
 
    for (int i = 0; i < num_row; i++)
    {
        float diff = std::abs(host_res[i] - host_Y[i]);
        EXPECT_LT(diff, 0.01);
    }

    hcsparseTeardown();

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
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);
}
