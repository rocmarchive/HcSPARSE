#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>

int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gA;
    hcdenseVector gX;
    hcdenseVector gB;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    if (argc != 2)
    {
        std::cout<<"Required mtx input file"<<std::endl;
        return 0;
    }

    const char* filename = argv[1];

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    } 

    double *host_X = (double*) calloc(num_col, sizeof(double));
    double *host_B = (double*) calloc(num_row, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_col; i++)
    {
       host_X[i] = 0;
    } 

    for (int i = 0; i < num_row; i++)
    {
        host_B[i] = rand()%100;
    }

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gA);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gB);

    gX.values = am_alloc(sizeof(double) * num_col, acc[1], 0);
    gB.values = am_alloc(sizeof(double) * num_row, acc[1], 0);

    control.accl_view.copy(host_X, gX.values, sizeof(double) * num_col);
    control.accl_view.copy(host_B, gB.values, sizeof(double) * num_row);

    gX.offValues = 0;
    gB.offValues = 0;

    gX.num_values = num_col;
    gB.num_values = num_row;

    gA.offValues = 0;
    gA.offColInd = 0;
    gA.offRowOff = 0;

    double *values = (double*)calloc(num_nonzero, sizeof(double));
    int *rowOffsets = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gA.values = am_alloc(sizeof(double) * num_nonzero, acc[1], 0);
    gA.rowOffsets = am_alloc(sizeof(int) * (num_row+1), acc[1], 0);
    gA.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseDCsrMatrixfromFile(&gA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }

    control.accl_view.copy(gA.values, values, sizeof(double) * num_nonzero);
    control.accl_view.copy(gA.rowOffsets, rowOffsets, sizeof(int) * (num_row+1));
    control.accl_view.copy(gA.colIndices, colIndices, sizeof(int) * num_nonzero);

    int maxIter = 1000;
    double relTol = 0.0001;
    double absTol = 0.0001;

    hcsparseSolverControl *solver_control;

    solver_control = hcsparseCreateSolverControl(DIAGONAL, maxIter, relTol, absTol); 

    hcsparseDcsrcg(&gX, &gA, &gB, solver_control, &control); 

    hcsparseTeardown();

    free(host_X);
    free(host_B);
    free(values);
    free(rowOffsets);
    free(colIndices);
    am_free(gX.values);
    am_free(gB.values);
    am_free(gA.values);
    am_free(gA.rowOffsets);
    am_free(gA.colIndices);

    return 0; 
}
