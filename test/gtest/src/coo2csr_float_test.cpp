#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(coo2csr_float_test, func_check)
{
    hcsparseCsrMatrix gCsrMat_res;
    hcsparseCsrMatrix gCsrMat_ref;
    hcsparseCooMatrix gCooMat;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view());

    hcsparseControl control(accl_view);

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat_res);
    hcsparseInitCsrMatrix(&gCsrMat_ref);
    hcsparseInitCooMatrix(&gCooMat);

    gCsrMat_res.offValues = 0;
    gCsrMat_res.offColInd = 0;
    gCsrMat_res.offRowOff = 0;

    gCsrMat_ref.offValues = 0;
    gCsrMat_ref.offColInd = 0;
    gCsrMat_ref.offRowOff = 0;

    gCooMat.offValues = 0;
    gCooMat.offColInd = 0;
    gCooMat.offRowInd = 0;

    const char* filename = "../../../src/input.mtx";

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
    }

    float *csr_res_values = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_res_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_csr_res_values(num_nonzero, csr_res_values);
    array_view<int> av_csr_res_rowOff(num_row+1, csr_res_rowOff);
    array_view<int> av_csr_res_colIndices(num_nonzero, csr_res_colIndices);

    gCsrMat_res.values = &av_csr_res_values;
    gCsrMat_res.rowOffsets = &av_csr_res_rowOff;
    gCsrMat_res.colIndices = &av_csr_res_colIndices;

    float *csr_ref_values = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_ref_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_ref_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_csr_ref_values(num_nonzero, csr_ref_values);
    array_view<int> av_csr_ref_rowOff(num_row+1, csr_ref_rowOff);
    array_view<int> av_csr_ref_colIndices(num_nonzero, csr_ref_colIndices);

    gCsrMat_ref.values = &av_csr_ref_values;
    gCsrMat_ref.rowOffsets = &av_csr_ref_rowOff;
    gCsrMat_ref.colIndices = &av_csr_ref_colIndices;

    float *coo_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_coo_values(num_nonzero, coo_values);
    array_view<int> av_coo_rowIndices(num_nonzero, coo_rowIndices);
    array_view<int> av_coo_colIndices(num_nonzero, coo_colIndices);

    gCooMat.values = &av_coo_values;
    gCooMat.rowIndices = &av_coo_rowIndices;
    gCooMat.colIndices = &av_coo_colIndices;

    hcsparseSCooMatrixfromFile(&gCooMat, filename, &control, false);

    hcsparseSCsrMatrixfromFile(&gCsrMat_ref, filename, &control, false);

    hcsparseScoo2csr(&gCooMat, &gCsrMat_res, &control);

    array_view<float> *av_ref_val = static_cast<array_view<float> *>(gCsrMat_ref.values);
    array_view<int> *av_ref_col = static_cast<array_view<int> *>(gCsrMat_ref.colIndices);
    array_view<int> *av_ref_row = static_cast<array_view<int> *>(gCsrMat_ref.rowOffsets);

    array_view<float> *av_res_val = static_cast<array_view<float> *>(gCsrMat_res.values);
    array_view<int> *av_res_col = static_cast<array_view<int> *>(gCsrMat_res.colIndices);
    array_view<int> *av_res_row = static_cast<array_view<int> *>(gCsrMat_res.rowOffsets);

    bool ispassed = 1;

    for (int i = 0; i < gCsrMat_res.num_nonzeros; i++)
    {
        int diff = std::abs((*av_ref_val)[i] - (*av_res_val)[i]);
        EXPECT_LT(diff, TOLERANCE);
    }

    for (int i = 0; i < gCsrMat_res.num_nonzeros; i++)
    {
        EXPECT_EQ((*av_ref_col)[i], (*av_res_col)[i]);
    }

    for (int i = 0; i < gCsrMat_res.num_rows+1; i++)
    {
        EXPECT_EQ((*av_ref_row)[i], (*av_res_row)[i]);
    }

    av_csr_res_values.synchronize();
    av_csr_res_rowOff.synchronize();
    av_csr_res_colIndices.synchronize();
    av_csr_ref_values.synchronize();
    av_csr_ref_rowOff.synchronize();
    av_csr_ref_colIndices.synchronize();
    av_coo_values.synchronize();
    av_coo_rowIndices.synchronize();
    av_coo_colIndices.synchronize();

    hcsparseTeardown();

    free(csr_res_values);
    free(csr_res_rowOff);
    free(csr_res_colIndices);
    free(csr_ref_values);
    free(csr_ref_rowOff);
    free(csr_ref_colIndices);
    free(coo_values);
    free(coo_rowIndices);
    free(coo_colIndices);
}
       
