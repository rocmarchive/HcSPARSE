#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gCsrMat;
    hcsparseCooMatrix gCooMat_ref;
    hcsparseCooMatrix gCooMat_res;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);
    hcsparseInitCooMatrix(&gCooMat_ref);
    hcsparseInitCooMatrix(&gCooMat_res);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    gCooMat_ref.offValues = 0;
    gCooMat_ref.offColInd = 0;
    gCooMat_ref.offRowInd = 0;

    gCooMat_res.offValues = 0;
    gCooMat_res.offColInd = 0;
    gCooMat_res.offRowInd = 0;

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

    float *coo_ref_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_ref_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_ref_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCooMat_ref.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCooMat_ref.rowIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);
    gCooMat_ref.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    float *coo_res_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_res_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_res_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCooMat_res.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCooMat_res.rowIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);
    gCooMat_res.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    hcsparseSCooMatrixfromFile(&gCooMat_ref, filename, &control, false);

    hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);

    hcsparseScsr2coo(&gCsrMat, &gCooMat_res, &control);

    control.accl_view.copy(gCooMat_ref.values, coo_ref_values, num_nonzero * sizeof(float));
    control.accl_view.copy(gCooMat_ref.rowIndices, coo_ref_rowIndices, num_nonzero * sizeof(int));
    control.accl_view.copy(gCooMat_ref.colIndices, coo_ref_colIndices, num_nonzero * sizeof(int));

    control.accl_view.copy(gCooMat_res.values, coo_res_values, num_nonzero * sizeof(float));
    control.accl_view.copy(gCooMat_res.rowIndices, coo_res_rowIndices, num_nonzero * sizeof(int));
    control.accl_view.copy(gCooMat_res.colIndices, coo_res_colIndices, num_nonzero * sizeof(int));

    bool ispassed = 1;

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        float diff = std::abs(coo_ref_values[i] - coo_res_values[i]);
        if (diff > 0.01)
        {
            std::cout<<"values "<<i<< " "<<coo_ref_values[i] <<" "<< coo_res_values[i]<<std::endl;
            ispassed = 0;
            break;
        }
    }

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        if (coo_ref_rowIndices[i] != coo_res_rowIndices[i])
        {
            std::cout<<"rowIndices "<<i<<" "<<coo_ref_rowIndices[i] << " " << coo_res_rowIndices[i]<<std::endl;
            ispassed = 0;
            break;
        }
    }

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        if (coo_ref_colIndices[i] != coo_res_colIndices[i])
        {
            std::cout<<"colIndices "<<i<<" "<<coo_ref_colIndices[i] << " " << coo_res_colIndices[i]<<std::endl;
            ispassed = 0;
            break;
        }
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    free(coo_ref_values);
    free(coo_ref_rowIndices);
    free(coo_ref_colIndices);
    free(coo_res_values);
    free(coo_res_rowIndices);
    free(coo_res_colIndices);
    am_free(gCsrMat.values);
    am_free(gCsrMat.rowOffsets);
    am_free(gCsrMat.colIndices);
    am_free(gCooMat_ref.values);
    am_free(gCooMat_ref.rowIndices);
    am_free(gCooMat_ref.colIndices);
    am_free(gCooMat_res.values);
    am_free(gCooMat_res.rowIndices);
    am_free(gCooMat_res.colIndices);

    return 0; 
}
