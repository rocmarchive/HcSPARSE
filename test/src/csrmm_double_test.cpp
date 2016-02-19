#include <hcsparse.h>
#include <iostream>
int main(int argc, char *argv[])
{
    hcsparseCsrMatrix gCsrMat;
    hcdenseMatrix gX;
    hcdenseMatrix gY;
    hcsparseScalar gAlpha;
    hcsparseScalar gBeta;

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

    int num_row_X, num_col_X, num_row_Y, num_col_Y;

    num_row_X = num_col_A;
    num_col_X = rand()%100;
    num_row_Y = num_row_A;
    num_col_Y = num_col_X;

    double *host_res = (double*) calloc(num_row_Y * num_col_Y, sizeof(double));
    double *host_X = (double*) calloc(num_row_X * num_col_X, sizeof(double));
    double *host_Y = (double*) calloc(num_row_Y * num_col_Y, sizeof(double));
    double *host_alpha = (double*) calloc(1, sizeof(double));
    double *host_beta = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_col_X * num_row_X; i++)
    {
       host_X[i] = rand()%100;
    } 

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        host_res[i] = host_Y[i] = rand()%100;
    }

    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    array_view<double> dev_X(num_col_X * num_row_X, host_X);
    array_view<double> dev_Y(num_row_Y * num_col_Y, host_Y);
    array_view<double> dev_alpha(1, host_alpha);
    array_view<double> dev_beta(1, host_beta);

    hcsparseSetup();
    hcsparseInitCsrMatrix(&gCsrMat);
    hcsparseInitScalar(&gAlpha);
    hcsparseInitScalar(&gBeta);
    hcdenseInitMatrix(&gX);
    hcdenseInitMatrix(&gY);

    gAlpha.value = &dev_alpha;
    gBeta.value = &dev_beta;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gAlpha.offValue = 0;
    gBeta.offValue = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gX.num_rows = num_row_X;
    gX.num_cols = num_col_X;
    gX.lead_dim = num_col_X;
    gY.num_rows = num_row_Y;
    gY.num_cols = num_col_Y;
    gY.lead_dim = num_col_Y;

    gX.major = rowMajor;
    gY.major = rowMajor;

    gCsrMat.offValues = 0;
    gCsrMat.offColInd = 0;
    gCsrMat.offRowOff = 0;

    double *values = (double*)calloc(num_nonzero, sizeof(double));
    int *rowIndices = (int*)calloc(num_row_A+1, sizeof(int));
    int *colIndices = (int*)calloc(num_nonzero, sizeof(int));

    array_view<double> av_values(num_nonzero, values);
    array_view<int> av_rowOff(num_row_A+1, rowIndices);
    array_view<int> av_colIndices(num_nonzero, colIndices);

    gCsrMat.values = &av_values;
    gCsrMat.rowOffsets = &av_rowOff;
    gCsrMat.colIndices = &av_colIndices;

    status = hcsparseDCsrMatrixfromFile(&gCsrMat, filename, &control, false);
   
    if (status != hcsparseSuccess)
    {
        std::cout<<"The input file should be in mtx format"<<std::endl;
        return 0;
    }
 
    hcsparseDcsrmm(&gAlpha, &gCsrMat, &gX, &gBeta, &gY, &control); 

    array_view<double> *av_val = static_cast<array_view<double> *>(gCsrMat.values);
    array_view<int> *av_row = static_cast<array_view<int> *>(gCsrMat.rowOffsets);
    array_view<int> *av_col = static_cast<array_view<int> *>(gCsrMat.colIndices);

    for (int col = 0; col < num_col_X; col++)
    {
        int indx = 0;
        for (int row = 0; row < num_row_A; row++)
        {
            host_res[row * num_col_Y + col] *= host_beta[0];
            for (; indx < (*av_row)[row+1]; indx++)
            {
                host_res[row * num_col_Y + col] = host_alpha[0] * host_X[(*av_col)[indx] * num_col_X + col] * (*av_val)[indx] + host_res[row * num_col_Y + col];
            }
        }
    }

    array_view<double> *av_res = static_cast<array_view<double> *>(gY.values);

    bool isPassed = 1;  

    for (int i = 0; i < num_row_Y * num_col_Y; i++)
    {
        double diff = std::abs(host_res[i] - (*av_res)[i]);
        if (diff > 0.01)
        {
            isPassed = 0;
            break;
        }
    }

    std::cout << (isPassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
