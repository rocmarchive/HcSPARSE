#include <hcsparse.h>
#include <iostream>
int main()
{
    hcsparseScalar gAlpha;
    hcsparseScalar gBeta;
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    double *host_R = (double*) calloc(num_elements, sizeof(double));
    double *host_res = (double*) calloc(num_elements, sizeof(double));
    double *host_X = (double*) calloc(num_elements, sizeof(double));
    double *host_Y = (double*) calloc(num_elements, sizeof(double));
    double *host_alpha = (double*) calloc(1, sizeof(double));
    double *host_beta = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    array_view<double> dev_R(num_elements, host_R);
    array_view<double> dev_X(num_elements, host_X);
    array_view<double> dev_Y(num_elements, host_Y);
    array_view<double> dev_alpha(1, host_alpha);
    array_view<double> dev_beta(1, host_beta);

    hcsparseSetup();
    hcsparseInitScalar(&gAlpha);
    hcsparseInitScalar(&gBeta);
    hcsparseInitVector(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gAlpha.value = &dev_alpha;
    gBeta.value = &dev_beta;
    gR.values = &dev_R;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gAlpha.offValue = 0;
    gBeta.offValue = 0;
    gR.offValues = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gR.num_values = num_elements;
    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseDaxpby(&gR, &gAlpha, &gX, &gBeta, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[i] = host_alpha[0] * host_X[i] + host_beta[0] * host_Y[i];
    }

    bool ispassed = 1;
    array_view<double> *av_res = static_cast<array_view<double> *>(gR.values);
    for (int i = 0; i < num_elements; i++)
    {
        if (host_res[i] != (*av_res)[i])
        {
            ispassed = 0;
            break;
        }
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
