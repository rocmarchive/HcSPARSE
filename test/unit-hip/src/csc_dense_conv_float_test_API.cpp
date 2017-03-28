#include "hip/hip_runtime.h"
#include "hipsparse.h"
#include <iostream>
#include "hc_am.hpp"

int main(int argc, char *argv[])
{
#if 0
    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();

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

    hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);
#else

    int num_row = 4;
    int num_col = 5;
    int num_nonzero = 9;

    float* valPtr = (float*)calloc(num_row*num_col, sizeof(float));
    int* colPtr = (int*)calloc(num_col+1, sizeof(int));
    int* rowInd = (int*)calloc(num_nonzero, sizeof(int));

    valPtr[0] = 1;
    valPtr[1] = 5;
    valPtr[2] = 4;
    valPtr[3] = 2;
    valPtr[4] = 3;
    valPtr[5] = 9;
    valPtr[6] = 7;
    valPtr[7] = 8;
    valPtr[8] = 6;

    colPtr[0] = 0;
    colPtr[1] = 2;
    colPtr[2] = 4;
    colPtr[3] = 6;
    colPtr[4] = 7;
    colPtr[5] = 9;

    rowInd[0] = 0;
    rowInd[1] = 2;
    rowInd[2] = 0;
    rowInd[3] = 1;
    rowInd[4] = 1;
    rowInd[5] = 3;
    rowInd[6] = 2;
    rowInd[7] = 2;
    rowInd[8] = 3;
#endif

     /* Test New APIs */
    hipsparseHandle_t handle;
    hipsparseStatus_t status1;

    status1 = hipsparseCreate(&handle);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      return -1;
    }
    std::cout << "Successfully initialized sparse library"<<std::endl;

    hipsparseMatDescr_t descrA;

    status1 = hipsparseCreateMatDescr(&descrA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    std::cout << "successfully created mat descriptor"<<std::endl;

    float* cscValA = (float*) am_alloc(num_nonzero * sizeof(float), handle->currentAccl, 0);
    int* cscColPtrA = (int*) am_alloc((num_col+1) * sizeof(int), handle->currentAccl, 0);
    int* cscRowIndA = (int*) am_alloc(num_nonzero * sizeof(int), handle->currentAccl, 0);
    float* A = (float*) am_alloc(num_row*num_col * sizeof(float), handle->currentAccl, 0);

    float *csc_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_colPtr = (int*)calloc(num_col+1, sizeof(int));
    int *csc_rowInd = (int*)calloc(num_nonzero, sizeof(int));    

    float *csc_res_val = (float*)calloc(num_nonzero, sizeof(float));
    int *csc_res_colPtr = (int*)calloc(num_col+1, sizeof(int));
    int *csc_res_rowInd = (int*)calloc(num_nonzero, sizeof(int));    
     
    handle->currentAcclView.copy(valPtr, cscValA, num_nonzero * sizeof(double));
    handle->currentAcclView.copy(colPtr, cscColPtrA, (num_col+1) * sizeof(int));
    handle->currentAcclView.copy(rowInd, cscRowIndA, num_nonzero * sizeof(int));

    status1 = hipsparseScsc2dense(handle, num_row, num_col,
                                 descrA, cscValA, cscColPtrA, cscRowIndA, A, num_col);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csc2dense conversion "<<std::endl;
      return -1;
    }
    std::cout << "csc2dense conv. - success"<<std::endl;

    handle->currentAcclView.copy(cscValA, csc_val, num_nonzero * sizeof(float));
    handle->currentAcclView.copy(cscColPtrA, csc_colPtr, (num_col+1) * sizeof(int));
    handle->currentAcclView.copy(cscRowIndA, csc_rowInd, num_nonzero * sizeof(int));

#if 0
    float* A_h = (float*) calloc(num_row*num_col, sizeof(float));
    handle->currentAcclView.copy(A, A_h, num_row*num_col * sizeof(float));
    std::cout << "After conversion - csrval" << std::endl;
    for (int i = 0; i < num_nonzero; i++)
      std::cout << i << " " << csc_val[i] <<  std::endl;

    std::cout << "After conversion - dense A " << std::endl;
    for (int i = 0; i < num_row*num_col; i++)
      std::cout << i << " " << A_h[i] <<  std::endl;
#endif


    int nnzperrow = 0;
    status1 = hipsparseSdense2csc(handle, num_row, num_col,
                                 descrA, A, num_row, &nnzperrow, cscValA, cscColPtrA, cscRowIndA);
    if (status1 != HIPSPARSE_STATUS_SUCCESS) {
      std::cout << "Error dense2csc conversion "<<std::endl;
      return -1;
    }
    std::cout << "dense2csc conv. - success"<<std::endl;

    handle->currentAcclView.copy(cscValA, csc_res_val, num_nonzero * sizeof(float));
    handle->currentAcclView.copy(cscColPtrA, csc_res_colPtr, (num_col+1) * sizeof(int));
    handle->currentAcclView.copy(cscRowIndA, csc_res_rowInd, num_nonzero * sizeof(int));

    bool ispassed = 1;

    std::cout <<"Val check" << std::endl;
    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::abs(csc_val[i] - csc_res_val[i]);
        if (diff > 0.01)
        {
            std::cout << i << " " << csc_val[i] << " " << csc_res_val[i] << std::endl;
            ispassed = 0;
            break;
        }
    }

    std::cout <<"rowind check" << std::endl;
    for (int i = 0; i < num_nonzero; i++)
    {
        if (csc_rowInd[i] != csc_res_rowInd[i])
        {
            std::cout << i << " " << csc_rowInd[i] << " " << csc_res_rowInd[i] << std::endl;
            ispassed = 0;
            break;
        }
    }

    std::cout <<"col check" << std::endl;
    for (int i = 0; i < num_col+1; i++)
    {
        if (csc_colPtr[i] != csc_res_colPtr[i])
        {
            std::cout << i << " " << csc_colPtr[i] << " " << csc_res_colPtr[i] << std::endl;
            ispassed = 0;
            break;
        }
    }
    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    free(csc_val);
    free(csc_colPtr);
    free(csc_rowInd);
    free(csc_res_val);
    free(csc_res_colPtr);
    free(csc_res_rowInd);
    am_free(cscValA);
    am_free(cscColPtrA);
    am_free(cscRowIndA);

    return 0;
}
