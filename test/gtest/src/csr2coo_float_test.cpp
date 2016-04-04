#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(csr2coo_float_test, func_check)
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

    const char* filename = "../../../../test/gtest/src/input.mtx";

    int num_nonzero, num_row, num_col;

    hcsparseStatus status;

    status = hcsparseHeaderfromFile(&num_nonzero, &num_row, &num_col, filename);

    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
    }

    float *csr_values = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_csr_values(num_nonzero, csr_values);
    array_view<int> av_csr_rowOff(num_row+1, csr_rowOff);
    array_view<int> av_csr_colIndices(num_nonzero, csr_colIndices);

    gCsrMat.values = &av_csr_values;
    gCsrMat.rowOffsets = &av_csr_rowOff;
    gCsrMat.colIndices = &av_csr_colIndices;

    float *coo_ref_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_ref_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_ref_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_coo_ref_values(num_nonzero, coo_ref_values);
    array_view<int> av_coo_ref_rowIndices(num_nonzero, coo_ref_rowIndices);
    array_view<int> av_coo_ref_colIndices(num_nonzero, coo_ref_colIndices);

    gCooMat_ref.values = &av_coo_ref_values;
    gCooMat_ref.rowIndices = &av_coo_ref_rowIndices;
    gCooMat_ref.colIndices = &av_coo_ref_colIndices;

    float *coo_res_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_res_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_res_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<float> av_coo_res_values(num_nonzero, coo_res_values);
    array_view<int> av_coo_res_rowIndices(num_nonzero, coo_res_rowIndices);
    array_view<int> av_coo_res_colIndices(num_nonzero, coo_res_colIndices);

    gCooMat_res.values = &av_coo_res_values;
    gCooMat_res.rowIndices = &av_coo_res_rowIndices;
    gCooMat_res.colIndices = &av_coo_res_colIndices;

    hcsparseSCooMatrixfromFile(&gCooMat_ref, filename, &control, false);

    hcsparseSCsrMatrixfromFile(&gCsrMat, filename, &control, false);

    hcsparseScsr2coo(&gCsrMat, &gCooMat_res, &control);

    array_view<float> *av_ref_val = static_cast<array_view<float> *>(gCooMat_ref.values);
    array_view<int> *av_ref_col = static_cast<array_view<int> *>(gCooMat_ref.colIndices);
    array_view<int> *av_ref_row = static_cast<array_view<int> *>(gCooMat_ref.rowIndices);

    array_view<float> *av_res_val = static_cast<array_view<float> *>(gCooMat_res.values);
    array_view<int> *av_res_col = static_cast<array_view<int> *>(gCooMat_res.colIndices);
    array_view<int> *av_res_row = static_cast<array_view<int> *>(gCooMat_res.rowIndices);

    bool ispassed = 1;

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        float diff = std::abs((*av_ref_val)[i] - (*av_res_val)[i]);
        EXPECT_LT(diff, TOLERANCE);
    }

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        EXPECT_EQ((*av_ref_col)[i], (*av_res_col)[i]);
    }

    for (int i = 0; i < gCooMat_res.num_nonzeros; i++)
    {
        EXPECT_EQ((*av_ref_row)[i], (*av_res_row)[i]);
    }

    av_csr_values.synchronize();
    av_csr_rowOff.synchronize();
    av_csr_colIndices.synchronize();
    av_coo_ref_values.synchronize();
    av_coo_ref_rowIndices.synchronize();
    av_coo_ref_colIndices.synchronize();
    av_coo_res_values.synchronize();
    av_coo_res_rowIndices.synchronize();
    av_coo_res_colIndices.synchronize();

    hcsparseTeardown();

    free(csr_values);
    free(csr_rowOff);
    free(csr_colIndices);
    free(coo_ref_values);
    free(coo_ref_rowIndices);
    free(coo_ref_colIndices);
    free(coo_res_values);
    free(coo_res_rowIndices);
    free(coo_res_colIndices);
}

