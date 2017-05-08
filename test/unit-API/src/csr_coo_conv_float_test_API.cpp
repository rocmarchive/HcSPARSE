#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gCsrMat;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

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

    gCsrMat.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCsrMat.rowOffsets = (int*) am_alloc((num_row+1) * sizeof(int), acc[1], 0);
    gCsrMat.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);

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

    int* csrRowPtrA = (int*) am_alloc((num_row+1) * sizeof(int), handle->currentAccl, 0);
    int* cooRowIndA = (int*) am_alloc((num_nonzero) * sizeof(int), handle->currentAccl, 0);

    int *csr_rowPtr = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_rowPtr = (int*)calloc(num_row+1, sizeof(int));

    control.accl_view.copy(gCsrMat.rowOffsets, csrRowPtrA, (num_row+1) * sizeof(int));

    status1 = hcsparseXcsr2coo(handle, csrRowPtrA, num_nonzero, num_row,
                               cooRowIndA, HCSPARSE_INDEX_BASE_ZERO);

    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error csr2coo conversion "<<std::endl;
      return -1;
    }
    std::cout << "csr2coo conv. - success"<<std::endl;

    control.accl_view.copy(csrRowPtrA, csr_rowPtr, (num_row+1) * sizeof(int));

    status1 = hcsparseXcoo2csr(handle, cooRowIndA, num_nonzero, num_row,
                               csrRowPtrA, HCSPARSE_INDEX_BASE_ZERO);

    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error coo2csr conversion "<<std::endl;
      return -1;
    }
    std::cout << "coo2csr conv. - success"<<std::endl;

    control.accl_view.copy(csrRowPtrA, csr_res_rowPtr, (num_row+1) * sizeof(int));

    bool ispassed = 1;

    for (int i = 0; i < num_row+1; i++)
    {
        if (csr_rowPtr[i] != csr_res_rowPtr[i])
        {
            std::cout << i << " " << csr_rowPtr[i] << " " << csr_res_rowPtr[i] << std::endl;
            ispassed = 0;
            break;
        }
    }
    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    free(csr_rowPtr);
    free(csr_res_rowPtr);
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);

    return 0;
}
