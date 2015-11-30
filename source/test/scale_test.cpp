#include <hcsparse.h>
#include <iostream>
int main()
{
    hcsparseScalar gAlpha;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<Concurrency::accelerator>acc = Concurrency::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_Y = (float*) calloc(num_elements, sizeof(float));
    float *host_alpha = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    host_alpha[0] = rand()%100;

    Concurrency::array_view<float> dev_X(num_elements, host_X);
    Concurrency::array_view<float> dev_Y(num_elements, host_Y);
    Concurrency::array_view<float> dev_alpha(1, host_alpha);

    hcsparseSetup();
    hcsparseInitScalar(&gAlpha);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gAlpha.value = &dev_alpha;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseSscale(&gX, &gAlpha, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = host_alpha[0] * host_Y[i];
    }

    bool ispassed = 1;
    Concurrency::array_view<float> *av_res = static_cast<Concurrency::array_view<float> *>(gX.values);
    for (int i = 0; i < num_elements; i++)
    {
        if (host_X[i] != (*av_res)[i])
        {
            ispassed = 0;
            break;
        }
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;
    
    return 0; 
}
