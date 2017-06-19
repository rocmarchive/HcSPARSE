#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main(int argc, char *argv[])
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

    gDenseMatA.values = am_alloc(sizeof(float) * num_row * num_col, acc[1], 0);
    gDenseMatB.values = am_alloc(sizeof(float) * num_row * num_col, acc[1], 0);
    gDenseMatC.values = am_alloc(sizeof(float) * num_row * num_col, acc[1], 0);

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

    gMatA.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gMatA.rowOffsets = am_alloc(sizeof(int) * (num_row+1), acc[1], 0);
    gMatA.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gMatA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    gMatB.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gMatB.rowOffsets = am_alloc(sizeof(int) * (num_row+1), acc[1], 0);
    gMatB.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    status = hcsparseSCsrMatrixfromFile(&gMatB, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    gMatC.values = am_alloc(sizeof(float) * num_nonzero, acc[1], 0);
    gMatC.rowOffsets = am_alloc(sizeof(int) * (num_row+1), acc[1], 0);
    gMatC.colIndices = am_alloc(sizeof(int) * num_nonzero, acc[1], 0);

    float *host_alpha = (float*) calloc(1, sizeof(float));
    float *host_beta = (float*) calloc(1, sizeof(float));

    float *gAlpha = am_alloc(sizeof(float) * 1, acc[1], 0);
    float *gBeta = am_alloc(sizeof(float) * 1, acc[1], 0);

    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    control.accl_view.copy(host_alpha, gAlpha, sizeof(float) * 1);
    control.accl_view.copy(host_beta, gBeta, sizeof(float) * 1);

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

    hcsparseMatDescr_t descrA, descrB, descrC;
    status1 = hcsparseCreateMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    status1 = hcsparseCreateMatDescr(&descrB);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    status1 = hcsparseCreateMatDescr(&descrC);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }

    status1 = hcsparseScsrgeam(handle, num_row, num_col,
                               gAlpha, descrA, num_nonzero, 
                               static_cast<const float *>(gMatA.values),
                               static_cast<const int *>(gMatA.rowOffsets),
                               static_cast<const int *>(gMatA.colIndices),
                               gBeta, descrB, num_nonzero,
                               static_cast<const float *>(gMatB.values),
                               static_cast<const int *>(gMatB.rowOffsets),
                               static_cast<const int *>(gMatB.colIndices),
                               descrC, (float *)gMatC.values,
                               (int *)gMatC.rowOffsets,
                               (int *)gMatC.colIndices);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error executing the module"<<std::endl;
      return -1;
    }

    gMatC.num_rows = num_row;
    gMatC.num_cols = num_col;

    hcsparseScsr2dense(&gMatA, &gDenseMatA, &control);
    hcsparseScsr2dense(&gMatB, &gDenseMatB, &control); 
    hcsparseScsr2dense(&gMatC, &gDenseMatC, &control); 
 
    float *dense_val_res = (float*) calloc(num_row*num_col, sizeof(float));
    float *denseMat_A = (float*)calloc(num_row*num_col, sizeof(float));
    float *denseMat_B = (float*)calloc(num_row*num_col, sizeof(float));
    float *denseMat_C = (float*)calloc(num_row*num_col, sizeof(float));
    
    control.accl_view.copy(gDenseMatA.values, denseMat_A, num_row * num_col * sizeof(float));
    control.accl_view.copy(gDenseMatB.values, denseMat_B, num_row * num_col * sizeof(float));
    control.accl_view.copy(gDenseMatC.values, denseMat_C, num_row * num_col * sizeof(float));
    
    for (int i = 0; i < num_row * num_col; i++)
    {
        dense_val_res[i] = denseMat_A[i] + denseMat_B[i];
    } 

    bool isPassed = 1;  

    for (int i = 0; i < num_row * num_col; i++)
    {
        float diff = std::abs(dense_val_res[i] - denseMat_C[i]);
        if (diff > 0.01)
        {
        std::cout<<i<<" "<<dense_val_res[i]<<" "<<denseMat_C[i]<< " "<<diff<<std::endl;
            isPassed = 0;
            break;
        }
    }

    std::cout << (isPassed?"TEST PASSED":"TEST FAILED") << std::endl;

    std::cout << "csrgeam Completed!" << std::endl;

    hcsparseTeardown();

    free(denseMat_A);
    free(denseMat_B);
    free(denseMat_C);
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

    return 0; 
}
