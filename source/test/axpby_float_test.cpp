#include <hcsparse.h>
#include <iostream>
int main()
{
    hcsparseScalar gAlpha;
    hcsparseScalar gBeta;
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<Concurrency::accelerator>acc = Concurrency::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    float *host_R = (float*) calloc(num_elements, sizeof(float));
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_Y = (float*) calloc(num_elements, sizeof(float));
    float *host_alpha = (float*) calloc(1, sizeof(float));
    float *host_beta = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    host_alpha[0] = rand()%100;
    host_beta[0] = rand()%100;

    Concurrency::array_view<float> dev_R(num_elements, host_R);
    Concurrency::array_view<float> dev_X(num_elements, host_X);
    Concurrency::array_view<float> dev_Y(num_elements, host_Y);
    Concurrency::array_view<float> dev_alpha(1, host_alpha);
    Concurrency::array_view<float> dev_beta(1, host_beta);

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

    gR.num_values = num_elements;
    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseSaxpby(&gR, &gAlpha, &gX, &gBeta, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = host_alpha[0] * host_X[i] + host_beta[0] * host_Y[i];
    }

    bool ispassed = 1;
    Concurrency::array_view<float> *av_res = static_cast<Concurrency::array_view<float> *>(gR.values);
    for (int i = 0; i < num_elements; i++)
    {
        if (host_R[i] != (*av_res)[i])
        {
            ispassed = 0;
            break;
        }
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
