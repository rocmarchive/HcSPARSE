#include <hcsparse.h>
#include <iostream>
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gCsrMat;
    hcsparseCsrMatrix gCsrMat_res;
    hcdenseMatrix gMat;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);
    hcsparseInitCsrMatrix(&gCsrMat_res);
    hcdenseInitMatrix(&gMat);

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    gCsrMat_res.offValues = 0;
    gCsrMat_res.offColInd = 0;
    gCsrMat_res.offRowOff = 0;

    gMat.offValues = 0;

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

    double *csr_values = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_colIndices = (int*)calloc(num_nonzero, sizeof(int));    

    array_view<double> av_csr_values(num_nonzero, csr_values);
    array_view<int> av_csr_rowOff(num_row+1, csr_rowOff);
    array_view<int> av_csr_colIndices(num_nonzero, csr_colIndices);

    gCsrMat.values = &av_csr_values;
    gCsrMat.rowOffsets = &av_csr_rowOff;
    gCsrMat.colIndices = &av_csr_colIndices;

    double *csr_res_values = (double*)calloc(num_nonzero, sizeof(double));
    int *csr_res_rowOff = (int*)calloc(num_row+1, sizeof(int));
    int *csr_res_colIndices = (int*)calloc(num_nonzero, sizeof(int));    

    array_view<double> av_csr_res_values(num_nonzero, csr_res_values);
    array_view<int> av_csr_res_rowOff(num_row+1, csr_res_rowOff);
    array_view<int> av_csr_res_colIndices(num_nonzero, csr_res_colIndices);

    gCsrMat_res.values = &av_csr_res_values;
    gCsrMat_res.rowOffsets = &av_csr_res_rowOff;
    gCsrMat_res.colIndices = &av_csr_res_colIndices;

    double *A_values = (double*)calloc(num_row*num_col, sizeof(double));

    array_view<double> av_A_values(num_row*num_col, A_values);

    gMat.values = &av_A_values;
    gMat.num_rows = num_row;
    gMat.num_cols = num_col;

    hcsparseDCsrMatrixfromFile(&gCsrMat, filename, &control, false);

    hcsparseDcsr2dense(&gCsrMat, &gMat, &control);

    hcsparseDdense2csr(&gMat, &gCsrMat_res, &control);

    bool ispassed = 1;

    for (int i = 0; i < num_nonzero; i++)
    {
        float diff = std::abs(av_csr_values[i] - av_csr_res_values[i]);
        if (diff > 0.1)
        {
            std::cout << i << " " << av_csr_values[i] << " " << av_csr_res_values[i] << std::endl;
            ispassed = 0;
            break;
        }
    }

    for (int i = 0; i < num_nonzero; i++)
    {
        if (av_csr_colIndices[i] != av_csr_res_colIndices[i])
        {
            std::cout << i << " " << av_csr_colIndices[i] << " " << av_csr_res_colIndices[i] << std::endl;
            ispassed = 0;
            break;
        }
    }

    for (int i = 0; i < num_row+1; i++)
    {
        if (av_csr_rowOff[i] != av_csr_res_rowOff[i])
        {
            std::cout << i << " " << av_csr_rowOff[i] << " " << av_csr_res_rowOff[i] << std::endl;
            ispassed = 0;
            break;
        }
    }
    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
