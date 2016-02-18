#include <hcsparse.h>
#include <iostream>
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gMatA;
    hcsparseCsrMatrix gMatB;
    hcsparseCsrMatrix gMatC;

    hcdenseMatrix gDenseMatA;
    hcdenseMatrix gDenseMatB;
    hcdenseMatrix gDenseMatC;

    std::vector<Concurrency::accelerator>acc = Concurrency::accelerator::get_all();
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

    Concurrency::array_view<float> av_dense_val_A(num_row*num_col, dense_val_A);
    Concurrency::array_view<float> av_dense_val_B(num_row*num_col, dense_val_B);
    Concurrency::array_view<float> av_dense_val_C(num_row*num_col, dense_val_C);

    gDenseMatA.values = &av_dense_val_A;
    gDenseMatB.values = &av_dense_val_B;
    gDenseMatC.values = &av_dense_val_C;

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

    float *values_A = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_A = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_A = (int*)calloc(num_nonzero, sizeof(int));

    Concurrency::array_view<float> av_values_A(num_nonzero, values_A);
    Concurrency::array_view<int> av_rowOff_A(num_row+1, rowIndices_A);
    Concurrency::array_view<int> av_colIndices_A(num_nonzero, colIndices_A);

    gMatA.values = &av_values_A;
    gMatA.rowOffsets = &av_rowOff_A;
    gMatA.colIndices = &av_colIndices_A;

    status = hcsparseSCsrMatrixfromFile(&gMatA, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    float *values_B = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_B = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_B = (int*)calloc(num_nonzero, sizeof(int));

    Concurrency::array_view<float> av_values_B(num_nonzero, values_B);
    Concurrency::array_view<int> av_rowOff_B(num_row+1, rowIndices_B);
    Concurrency::array_view<int> av_colIndices_B(num_nonzero, colIndices_B);

    gMatB.values = &av_values_B;
    gMatB.rowOffsets = &av_rowOff_B;
    gMatB.colIndices = &av_colIndices_B;

    status = hcsparseSCsrMatrixfromFile(&gMatB, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    float *values_C = (float*)calloc(num_nonzero, sizeof(float));
    int *rowIndices_C = (int*)calloc(num_row+1, sizeof(int));
    int *colIndices_C = (int*)calloc(num_nonzero, sizeof(int));

    Concurrency::array_view<float> av_values_C(num_nonzero, values_C);
    Concurrency::array_view<int> av_rowOff_C(num_row+1, rowIndices_C);
    Concurrency::array_view<int> av_colIndices_C(num_nonzero, colIndices_C);

    gMatB.values = &av_values_C;
    gMatB.rowOffsets = &av_rowOff_C;
    gMatB.colIndices = &av_colIndices_C;

    hcsparseScsrSpGemm(&gMatA, &gMatB, &gMatC, &control);

/*    hcsparseScsr2dense(&gMatA, &gDenseMatA, &control);
    hcsparseScsr2dense(&gMatB, &gDenseMatB, &control); 
    hcsparseScsr2dense(&gMatC, &gDenseMatC, &control); 
 
    float *dense_val_res = (float*) calloc(num_row*num_col, sizeof(float));

    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
        {
            for (int k = 0; k < num_col; k++)
            {
                dense_val_res[i * num_col + j] = av_dense_val_A[i * num_col + k] * av_dense_val_B[k * num_col + j];
            }
        }
    } 

    bool isPassed = 1;  

    for (int i = 0; i < num_row * num_col; i++)
    {
        float diff = std::abs(dense_val_res[i] - av_dense_val_C[i]);
        if (diff > 0.01)
        {
        std::cout<<i<<" "<<dense_val_res[i]<<" "<<av_dense_val_C[i]<< " "<<diff<<std::endl;
            isPassed = 0;
            break;
        }
    }

    std::cout << (isPassed?"TEST PASSED":"TEST FAILED") << std::endl;
*/
    std::cout << "csrSpGemm Completed!" << std::endl;

    hcsparseTeardown();

    return 0; 
}
