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

    float *host_res = (float*) calloc(num_row_Y * num_col_Y, sizeof(float));
    float *host_X = (float*) calloc(num_row_X * num_col_X, sizeof(float));
    float *host_Y = (float*) calloc(num_row_Y * num_col_Y, sizeof(float));
    float *host_alpha = (float*) calloc(1, sizeof(float));
    float *host_beta = (float*) calloc(1, sizeof(float));

    hcsparseCsrMatrix gCsrMat;
    float *gX;
    float *gY;
    float *gAlpha;
    float *gBeta;

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);

    gX = am_alloc(sizeof(float) * num_col_X * num_row_X, acc[1], 0);
    gY = am_alloc(sizeof(float) * num_row_Y * num_col_Y, acc[1], 0);
    gAlpha = am_alloc(sizeof(float) * 1, acc[1], 0);
    gBeta = am_alloc(sizeof(float) * 1, acc[1], 0);

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

    control.accl_view.copy(host_X, gX, sizeof(float) * num_col_X * num_row_X);
    control.accl_view.copy(host_Y, gY, sizeof(float) * num_row_Y * num_col_Y);
    control.accl_view.copy(host_alpha, gAlpha, sizeof(float) * 1);
    control.accl_view.copy(host_beta, gBeta, sizeof(float) * 1);

    float *values = (float*)calloc(num_nonzero, sizeof(float));
    int *rowOffsets = (int*)calloc(num_row_A+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCsrMat.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gCsrMat.rowOffsets = am_alloc(sizeof(int) * (num_row_A+1), acc[1], 0);
    gCsrMat.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }

    control.accl_view.copy(gCsrMat.values, values, sizeof(float) * num_nonzero);
    control.accl_view.copy(gCsrMat.rowOffsets, rowOffsets, sizeof(int) * (num_row_A+1));
    control.accl_view.copy(gCsrMat.colIndices, colIndices, sizeof(int) * num_nonzero);

    hcsparseOperation_t transA = HCSPARSE_OPERATION_NON_TRANSPOSE;
    int nnz = 0;

    status1 = hcsparseScsrmm(handle, transA, num_row_A, num_col_Y,
                            num_col_A, nnz, static_cast<const float*>(gAlpha), descrA,
                            static_cast<const float*>(gCsrMat.values),
                            (int *) gCsrMat.rowOffsets,
                            (int *)gCsrMat.colIndices, (float*)gX, num_col_X,
                            static_cast<const float*>(gBeta), (float *)gY, num_col_Y);

    for (int col = 0; col < num_col_X; col++)
    {
        int indx = 0;
        for (int row = 0; row < num_row_A; row++)
        {
            float sum = 0.0;
            for (; indx < rowOffsets[row+1]; indx++)
            {
                sum += host_alpha[0] * host_X[colIndices[indx] * num_col_X + col] * values[indx];
            }
            host_res[row * num_col_Y + col] = sum + host_beta[0] * host_res[row * num_col_Y + col];
        }
    }

    control.accl_view.copy(gY, host_Y, sizeof(float) * num_row_Y * num_col_Y);

    bool isPassed = 1;

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        float diff = std::abs(host_res[i] - host_Y[i]);
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
