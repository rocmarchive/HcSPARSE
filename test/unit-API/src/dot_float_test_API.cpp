#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gCsrMat;
    hcsparseScalar gR;
    hcdenseVector gY;

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

    int num_elements = num_row_A * num_col_A;
    float *host_res = (float*) calloc(1, sizeof(float)); 	
    float *host_Y = (float*) calloc(num_elements, sizeof(float));
    float *host_X = (float*) calloc(num_nonzero, sizeof(float));
    float *host_R = (float*) calloc(1, sizeof(float));
    int *host_X_colInd = (int *)calloc(num_nonzero, sizeof(int));

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gY);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;
 
    gCsrMat.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gCsrMat.rowOffsets = am_alloc(sizeof(int) * (num_row_A+1), acc[1], 0);
    gCsrMat.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    gR.value = am_alloc(sizeof(float) * 1, acc[1], 0);
    gY.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_Y[i] = rand()%100;
    }
    
    control.accl_view.copy(gCsrMat.values, host_X, sizeof(float) * num_nonzero);
    control.accl_view.copy(gCsrMat.colIndices, host_X_colInd, sizeof(int) * num_nonzero);
    control.accl_view.copy(host_R, gR.value, sizeof(float) * 1);
    control.accl_view.copy(host_Y, gY.values, sizeof(float) * num_elements);

    gR.offValue = 0;
    gY.offValues = 0;
    gY.num_values = num_elements;

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

    hcsparseMatDescr_t descrA;

    status1 = hcsparseCreateMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }

    status1 = hcsparseSdoti(handle, num_nonzero, (const float *)gCsrMat.values,
                           (const int *)gCsrMat.colIndices, (float *)gY.values,
                           (float *)gR.value, HCSPARSE_INDEX_BASE_ZERO);
 
    for (int i = 0; i < num_nonzero; i++)
    {
        host_res[0] += host_X[i] * host_Y[host_X_colInd[i]];
    }

    bool ispassed = 1;

    control.accl_view.copy(gR.value, host_R, sizeof(float) * 1);

    if (host_res[0] != host_R[0])
    {
        ispassed = 0;
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();
    status1 = hcsparseDestroyMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error clearing mat descrptr"<<std::endl;
      return -1;
    }

    status1 = hcsparseDestroy(&handle);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
      return -1;
    }
 
    free(host_res);
    free(host_Y);
    free(host_R);
    free(host_X);
    free(host_X_colInd);
    am_free(gR.value);
    am_free(gY.values);
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);

    return 0; 
}
