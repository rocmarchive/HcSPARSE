#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>
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

    const char* filename = "./../../../../test/gtest/src/input.mtx";

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

    gCsrMat_res.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCsrMat_res.rowOffsets = (int*) am_alloc((num_row+1) * sizeof(int), acc[1], 0);
    gCsrMat_res.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    float *csr_ref_values = (float*)calloc(num_nonzero, sizeof(float));
    int *csr_ref_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_ref_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCsrMat_ref.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCsrMat_ref.rowOffsets = (int*) am_alloc((num_row+1) * sizeof(int), acc[1], 0);
    gCsrMat_ref.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    gCooMat.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCooMat.rowIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);
    gCooMat.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    float *coo_values = (float*)calloc(num_nonzero, sizeof(float));
    int *coo_rowIndices = (int*)calloc(num_nonzero, sizeof(int));
    int *coo_colIndices = (int*)calloc(num_nonzero, sizeof(int));

    gCooMat.values = (float*) am_alloc(num_nonzero * sizeof(float), acc[1], 0);
    gCooMat.rowIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);
    gCooMat.colIndices = (int*) am_alloc(num_nonzero * sizeof(int), acc[1], 0);

    hcsparseSCooMatrixfromFile(&gCooMat, filename, &control, false);

    hcsparseSCsrMatrixfromFile(&gCsrMat_ref, filename, &control, false);

    hcsparseScoo2csr(&gCooMat, &gCsrMat_res, &control);
    control.accl_view.wait();

    control.accl_view.copy(gCsrMat_ref.values, csr_ref_values, num_nonzero * sizeof(float));
    control.accl_view.copy(gCsrMat_ref.rowOffsets, csr_ref_rowOff, (num_row+1) * sizeof(int));
    control.accl_view.copy(gCsrMat_ref.colIndices, csr_ref_colIndices, num_nonzero * sizeof(int));

    control.accl_view.copy(gCsrMat_res.values, csr_res_values, num_nonzero * sizeof(float));
    control.accl_view.copy(gCsrMat_res.rowOffsets, csr_res_rowOff, (num_row+1) * sizeof(int));
    control.accl_view.copy(gCsrMat_res.colIndices, csr_res_colIndices, num_nonzero * sizeof(int));
    bool ispassed = 1;

    for (int i = 0; i < gCsrMat_res.num_nonzeros; i++)
    {
        int diff = std::abs(csr_ref_values[i] - csr_res_values[i]);
        EXPECT_LT(diff, TOLERANCE);
    }

    for (int i = 0; i < gCsrMat_res.num_nonzeros; i++)
    {
        EXPECT_EQ(csr_ref_colIndices[i], csr_res_colIndices[i]);
    }

    for (int i = 0; i < gCsrMat_res.num_rows+1; i++)
    {
        EXPECT_EQ(csr_ref_rowOff[i], csr_res_rowOff[i]);
    }

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
    am_free(gCsrMat_res.values);
    am_free(gCsrMat_res.rowOffsets);
    am_free(gCsrMat_res.colIndices);
    am_free(gCsrMat_ref.values);
    am_free(gCsrMat_ref.rowOffsets);
    am_free(gCsrMat_ref.colIndices);
    am_free(gCooMat.values);
    am_free(gCooMat.rowIndices);
    am_free(gCooMat.colIndices);

}
       
