#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>
#include "gtest/gtest.h"

TEST(spcsrmm_float_test, func_check)
{
    hcsparseCsrMatrix gMatA;
    hcsparseCsrMatrix gMatB;
    hcsparseCsrMatrix gMatC;

    hcdenseMatrix gDenseMatA;
    hcdenseMatrix gDenseMatB;
    hcdenseMatrix gDenseMatC;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    const char* filename = "./../../../../test/gtest/src/input.mtx";

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    } 

    hcsparseSetup();

    hcdenseInitMatrix(&gDenseMatA);
    hcdenseInitMatrix(&gDenseMatB);
    hcdenseInitMatrix(&gDenseMatC);

    hcsparseInitCsrMatrix(&gMatA);
    hcsparseInitCsrMatrix(&gMatB);
    hcsparseInitCsrMatrix(&gMatC);

    float *dense_val_A = (float*) calloc(num_row*num_col, sizeof(float));
    float *dense_val_B = (float*) calloc(num_row*num_col, sizeof(float));
    float *dense_val_C = (float*) calloc(num_row*num_col, sizeof(float));

    gDenseMatA.values = am_alloc(sizeof(float)*num_row*num_col, acc[1], 0);
    gDenseMatB.values = am_alloc(sizeof(float)*num_row*num_col, acc[1], 0);
    gDenseMatC.values = am_alloc(sizeof(float)*num_row*num_col, acc[1], 0);

    gDenseMatA.num_rows = num_row;
    gDenseMatA.num_cols = num_col;
    gDenseMatB.num_rows = num_row;
    gDenseMatB.num_cols = num_col;
    gDenseMatC.num_rows = num_row;
    gDenseMatC.num_cols = num_col;

    gDenseMatA.offValues = 0;
    gDenseMatB.offValues = 0;
    gDenseMatC.offValues = 0;

    gMatA.offValues = 0;
    gMatA.offColInd = 0;
    gMatA.offRowOff = 0;

    gMatB.offValues = 0;
    gMatB.offValues = 0;
    gMatB.offColInd = 0;

    gMatC.offRowOff = 0;
    gMatC.offColInd = 0;
    gMatC.offRowOff = 0;

    float *values_A = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_A = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_A = (int*)calloc(num_nonzero, sizeof(int));

    gMatA.values = am_alloc(sizeof(float)*num_nonzero, acc[1], 0);
    gMatA.rowOffsets = am_alloc(sizeof(float)*num_row+1, acc[1], 0);
    gMatA.colIndices = am_alloc(sizeof(float)*num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gMatA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }
 
    float *values_B = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_B = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_B = (int*)calloc(num_nonzero, sizeof(int));

    gMatB.values = am_alloc(sizeof(float)*num_nonzero, acc[1], 0);
    gMatB.rowOffsets = am_alloc(sizeof(float)*num_row+1, acc[1], 0);
    gMatB.colIndices = am_alloc(sizeof(float)*num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gMatB, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }
 
    float *values_C = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_C = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_C = (int*)calloc(num_nonzero, sizeof(int));

    gMatC.values = am_alloc(sizeof(float)*20736, acc[1], 0);
    gMatC.rowOffsets = am_alloc(sizeof(float)*num_row+1, acc[1], 0);
    gMatC.colIndices = am_alloc(sizeof(float)*20736, acc[1], 0);

    hcsparseScsrSpGemm(&gMatA, &gMatB, &gMatC, &control);
    control.accl_view.wait();

/*    hcsparseScsr2dense(&gMatA, &gDenseMatA, &control);
    hcsparseScsr2dense(&gMatB, &gDenseMatB, &control); 
    hcsparseScsr2dense(&gMatC, &gDenseMatC, &control); 
 
    float *dense_val_res = (float*) calloc(num_row*num_col, sizeof(float));

    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
        {
            for (int k = 0; k < num_col; k++)
            {
                dense_val_res[i * num_col + j] = av_dense_val_A[i * num_col + k] * av_dense_val_B[k * num_col + j];
            }
        }
    } 

    bool isPassed = 1;  

    for (int i = 0; i < num_row * num_col; i++)
    {
        float diff = std::abs(dense_val_res[i] - av_dense_val_C[i]);
        EXPECT_LT(diff, 0.01);
    }*/

    hcsparseTeardown();

    free(dense_val_A);
    free(dense_val_B);
    free(dense_val_C);
    free(values_A);
    free(rowIndices_A);
    free(colIndices_A);
    free(values_B);
    free(rowIndices_B);
    free(colIndices_B);
    free(values_C);
    free(rowIndices_C);
    free(colIndices_C);
    am_free(gDenseMatA.values);
    am_free(gDenseMatB.values);
    am_free(gDenseMatC.values);
    am_free(gMatA.values);
    am_free(gMatA.rowOffsets);
    am_free(gMatA.colIndices);
    am_free(gMatB.values);
    am_free(gMatB.rowOffsets);
    am_free(gMatB.colIndices);
    am_free(gMatC.values);
    am_free(gMatC.rowOffsets);
    am_free(gMatC.colIndices);
}
