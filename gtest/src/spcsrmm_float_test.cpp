#include <hcsparse.h>
#include <iostream>
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

    const char* filename = "../../../../../input_sparse/cant.mtx";

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

    array_view<float> av_dense_val_A(num_row*num_col, dense_val_A);
    array_view<float> av_dense_val_B(num_row*num_col, dense_val_B);
    array_view<float> av_dense_val_C(num_row*num_col, dense_val_C);

    gDenseMatA.values = &av_dense_val_A;
    gDenseMatB.values = &av_dense_val_B;
    gDenseMatC.values = &av_dense_val_C;

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

    array_view<float> av_values_A(num_nonzero, values_A);
    array_view<int> av_rowOff_A(num_row+1, rowIndices_A);
    array_view<int> av_colIndices_A(num_nonzero, colIndices_A);

    gMatA.values = &av_values_A;
    gMatA.rowOffsets = &av_rowOff_A;
    gMatA.colIndices = &av_colIndices_A;

    status = hcsparseSCsrMatrixfromFile(&gMatA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }
 
    float *values_B = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_B = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_B = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_values_B(num_nonzero, values_B);
    array_view<int> av_rowOff_B(num_row+1, rowIndices_B);
    array_view<int> av_colIndices_B(num_nonzero, colIndices_B);

    gMatB.values = &av_values_B;
    gMatB.rowOffsets = &av_rowOff_B;
    gMatB.colIndices = &av_colIndices_B;

    status = hcsparseSCsrMatrixfromFile(&gMatB, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }
 
    float *values_C = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_C = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_C = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_values_C(num_nonzero, values_C);
    array_view<int> av_rowOff_C(num_row+1, rowIndices_C);
    array_view<int> av_colIndices_C(num_nonzero, colIndices_C);

    gMatB.values = &av_values_C;
    gMatB.rowOffsets = &av_rowOff_C;
    gMatB.colIndices = &av_colIndices_C;

    hcsparseScsrSpGemm(&gMatA, &gMatB, &gMatC, &control);

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

    av_dense_val_A.synchronize();
    av_dense_val_B.synchronize();
    av_dense_val_C.synchronize();
    av_values_A.synchronize();
    av_rowOff_A.synchronize();
    av_colIndices_A.synchronize();
    av_values_B.synchronize();
    av_rowOff_B.synchronize();
    av_colIndices_B.synchronize();
    av_values_C.synchronize();
    av_rowOff_C.synchronize();
    av_colIndices_C.synchronize();

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
}
