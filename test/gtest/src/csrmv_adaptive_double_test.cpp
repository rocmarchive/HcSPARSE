#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

TEST(csrmv_adaptive_double_test, func_check)
{
    hcsparseCsrMatrix gCsrMat;
    hcdenseVector gX;
    hcdenseVector gY;
    hcsparseScalar gAlpha;
    hcsparseScalar gBeta;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    const char* filename = "../../../../test/gtest/src/input.mtx";

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    } 

    double *host_res = (double*) calloc(num_row, sizeof(double));
    double *host_X = (double*) calloc(num_col, sizeof(double));
    double *host_Y = (double*) calloc(num_row, sizeof(double));
    double *host_alpha = (double*) calloc(1, sizeof(double));
    double *host_beta = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_col; i++)
    {
       host_X[i] = 1;//rand()%100;
    } 

    for (int i = 0; i < num_row; i++)
    {
        host_res[i] = host_Y[i] = 1;//rand()%100;
    }

    host_alpha[0] = 1;
    host_beta[0] = 0;

    array_view<double> dev_X(num_col, host_X);
    array_view<double> dev_Y(num_row, host_Y);
    array_view<double> dev_alpha(1, host_alpha);
    array_view<double> dev_beta(1, host_beta);

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);
    hcsparseInitScalar(&gAlpha);
    hcsparseInitScalar(&gBeta);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gAlpha.value = &dev_alpha;
    gBeta.value = &dev_beta;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gAlpha.offValue = 0;
    gBeta.offValue = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gX.num_values = num_col;
    gY.num_values = num_row;

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    double *values = (double*)calloc(num_nonzero, sizeof(double));
    int *rowIndices = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));
    ulong *rowBlocks = (ulong*)calloc(num_nonzero, sizeof(ulong));

    array_view<double> av_values(num_nonzero, values);
    array_view<int> av_rowOff(num_row+1, rowIndices);
    array_view<int> av_colIndices(num_nonzero, colIndices);
    array_view<ulong> av_rowBlocks(num_nonzero, rowBlocks);

    gCsrMat.values = &av_values;
    gCsrMat.rowOffsets = &av_rowOff;
    gCsrMat.colIndices = &av_colIndices;
    gCsrMat.rowBlocks = &av_rowBlocks;

    status = hcsparseDCsrMatrixfromFile(&gCsrMat, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }

    hcsparseCsrMetaSize(&gCsrMat, &control);

    hcsparseCsrMetaCompute(&gCsrMat, &control);
 
    hcsparseDcsrmv(&gAlpha, &gCsrMat, &gX, &gBeta, &gY, &control); 

    array_view<double> *av_val = static_cast<array_view<double> *>(gCsrMat.values);
    array_view<int> *av_row = static_cast<array_view<int> *>(gCsrMat.rowOffsets);
    array_view<int> *av_col = static_cast<array_view<int> *>(gCsrMat.colIndices);

    int col = 0;
    for (int row = 0; row < num_row; row++)
    {
        host_res[row] *= host_beta[0];
        for (; col < (*av_row)[row+1]; col++)
        {
            host_res[row] = host_alpha[0] * host_X[(*av_col)[col]] * (*av_val)[col] + host_res[row];
        }
    }

    array_view<double> *av_res = static_cast<array_view<double> *>(gY.values);

    bool isPassed = 1;  
 
    for (int i = 0; i < num_row; i++)
    {
        double diff = std::abs(host_res[i] - (*av_res)[i]);
        EXPECT_LT(diff, 0.01);
    }

    dev_X.synchronize();
    dev_Y.synchronize();
    dev_alpha.synchronize();
    dev_beta.synchronize();
    av_values.synchronize();
    av_rowOff.synchronize();
    av_colIndices.synchronize();
    av_rowBlocks.synchronize();

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_Y);
    free(host_alpha);
    free(host_beta);
    free(values);
    free(rowIndices);
    free(colIndices);
    free(rowBlocks);
}
