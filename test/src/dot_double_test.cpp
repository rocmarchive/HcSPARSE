#include <hcsparse.h>
#include <iostream>
int main()
{
    hcsparseScalar gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 1000;
    double *host_res = (double*) calloc(1, sizeof(double));
    double *host_X = (double*) calloc(num_elements, sizeof(double));
    double *host_Y = (double*) calloc(num_elements, sizeof(double));
    double *host_R = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    array_view<double> dev_X(num_elements, host_X);
    array_view<double> dev_Y(num_elements, host_Y);
    array_view<double> dev_R(1, host_R);

    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gR.value = &dev_R;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gR.offValue = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseDdot(&gR, &gX, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i] * host_Y[i];
    }

    bool ispassed = 1;
    array_view<double> *av_res = static_cast<array_view<double> *>(gR.value);
    if (host_res[0] != (*av_res)[0])
    {
        ispassed = 0;
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
