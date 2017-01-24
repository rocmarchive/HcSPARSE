#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

TEST(bicgStab_diagonal_float_test, func_check)
{
    hcsparseCsrMatrix gA;
    hcdenseVector gX;
    hcdenseVector gB;

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
        exit (1);
    } 

    float *host_X = (float*) calloc(num_col, sizeof(float));
    float *host_B = (float*) calloc(num_row, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_col; i++)
    {
       host_X[i] = rand()%100;
    } 

    for (int i = 0; i < num_row; i++)
    {
        host_B[i] = rand()%100;
    }

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gA);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gB);

    gX.values = am_alloc(sizeof(float)*num_col, acc[1], 0);
    gB.values = am_alloc(sizeof(float)*num_row, acc[1], 0);

    gX.offValues = 0;
    gB.offValues = 0;

    gX.num_values = num_col;
    gB.num_values = num_row;

    gA.offValues = 0;
    gA.offColInd = 0;
    gA.offRowOff = 0;

    control.accl_view.copy(host_X, gX.values, sizeof(float) * num_col);
    control.accl_view.copy(host_B, gB.values, sizeof(float) * num_row);

    float *values = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gA.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gA.rowOffsets = am_alloc(sizeof(float) * num_row+1, acc[1], 0);
    gA.colIndices = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        exit(1);
    }

    int maxIter = 100;
    double relTol = 0.0001;
    double absTol = 0.0001;

    hcsparseSolverControl *solver_control;

    solver_control = hcsparseCreateSolverControl(DIAGONAL, maxIter, relTol, absTol); 

    hcsparseScsrbicgStab(&gX, &gA, &gB, solver_control, &control); 

    hcsparseTeardown();

    free(host_X);
    free(host_B);
    free(values);
    free(rowIndices);
    free(colIndices);
    am_free(gX.values);
    am_free(gB.values);
    am_free(gA.values);
    am_free(gA.rowOffsets);
    am_free(gA.colIndices);

}
