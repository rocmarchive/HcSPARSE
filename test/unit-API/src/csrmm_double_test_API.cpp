#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main(int argc, char *argv[])
{

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    if (argc != 2)
    {
        std::cout<<"Required mtx input file"<<std::endl;
        return 0;
    }

    const char* filename = argv[1];

    int num_nonzero, num_row_A, num_col_A;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row_A, &num_col_A, filename);
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    } 

     /* Test New APIs */
    hcsparseHandle_t handle;
    hcsparseStatus_t status1;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();

    status1 = hcsparseCreate(&handle, &av);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      return -1;
    }
    std::cout << "Successfully initialized sparse library"<<std::endl;

    hcsparseMatDescr_t descrA;

    status1 = hcsparseCreateMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    std::cout << "successfully created mat descriptor"<<std::endl;

    int num_row_X, num_col_X, num_row_Y, num_col_Y;

    num_row_X = num_col_A;
    num_col_X = rand()%100;
    num_row_Y = num_row_A;
    num_col_Y = num_col_X;

    double *host_res = (double*) calloc(num_row_Y * num_col_Y, sizeof(double));
    double *host_X = (double*) calloc(num_row_X * num_col_X, sizeof(double));
    double *host_Y = (double*) calloc(num_row_Y * num_col_Y, sizeof(double));
    double *host_alpha = (double*) calloc(1, sizeof(double));
    double *host_beta = (double*) calloc(1, sizeof(double));

    hcsparseCsrMatrix gCsrMat;
    double *gX;
    double *gY;
    double *gAlpha;
    double *gBeta;

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);

    gX = am_alloc(sizeof(double) * num_col_X * num_row_X, acc[1], 0);
    gY = am_alloc(sizeof(double) * num_row_Y * num_col_Y, acc[1], 0);
    gAlpha = am_alloc(sizeof(double) * 1, acc[1], 0);
    gBeta = am_alloc(sizeof(double) * 1, acc[1], 0);

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

    control.accl_view.copy(host_X, gX, sizeof(double) * num_col_X * num_row_X);
    control.accl_view.copy(host_Y, gY, sizeof(double) * num_row_Y * num_col_Y);
    control.accl_view.copy(host_alpha, gAlpha, sizeof(double) * 1);
    control.accl_view.copy(host_beta, gBeta, sizeof(double) * 1);

    double *values = (double*)calloc(num_nonzero, sizeof(double));
    int *rowOffsets = (int*)calloc(num_row_A+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCsrMat.values = am_alloc(sizeof(double) * num_nonzero, acc[1], 0);
    gCsrMat.rowOffsets = am_alloc(sizeof(int) * (num_row_A+1), acc[1], 0);
    gCsrMat.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseDCsrMatrixfromFile(&gCsrMat, filename, &control, false);
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }

    control.accl_view.copy(gCsrMat.values, values, sizeof(double) * num_nonzero);
    control.accl_view.copy(gCsrMat.rowOffsets, rowOffsets, sizeof(int) * (num_row_A+1));
    control.accl_view.copy(gCsrMat.colIndices, colIndices, sizeof(int) * num_nonzero);

    hcsparseOperation_t transA = HCSPARSE_OPERATION_NON_TRANSPOSE;
    int nnz = 0;

    status1 = hcsparseDcsrmm(handle, transA, num_row_A, num_col_Y,
                            num_col_A, nnz, static_cast<const double*>(gAlpha), descrA,
                            static_cast<const double*>(gCsrMat.values),
                            (int *) gCsrMat.rowOffsets,
                            (int *)gCsrMat.colIndices, (double*)gX, num_col_X,
                            static_cast<const double*>(gBeta), (double *)gY, num_col_Y);

    for (int col = 0; col < num_col_X; col++)
    {
        int indx = 0;
        for (int row = 0; row < num_row_A; row++)
        {
            double sum = 0.0;
            for (; indx < rowOffsets[row+1]; indx++)
            {
                sum += host_alpha[0] * host_X[colIndices[indx] * num_col_X + col] * values[indx];
            }
            host_res[row * num_col_Y + col] = sum + host_beta[0] * host_res[row * num_col_Y + col];
        }
    }

    control.accl_view.copy(gY, host_Y, sizeof(double) * num_row_Y * num_col_Y);

    bool isPassed = 1;

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        double diff = std::abs(host_res[i] - host_Y[i]);
        if (diff > 0.01)
        {
            std::cout<<i<<" "<<host_res[i]<<" "<<host_Y[i]<< " "<<diff<<std::endl;
            isPassed = 0;
            break;
        }
    }

    std::cout << (isPassed?"TEST PASSED":"TEST FAILED") << std::endl;

    status1 = hcsparseDestroyMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    std::cout << "successfully created mat descriptor"<<std::endl;

    status1 = hcsparseDestroy(&handle);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
      return -1;
    }
    std::cout << "Successfully deinitialized sparse library"<<std::endl;
   
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
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);


    return 0;
}
