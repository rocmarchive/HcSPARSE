#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main()
{
    hcsparseScalar gR;
    hcdenseVector gX;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 10000;
    double *host_res = (double*) calloc(1, sizeof(double));
    double *host_X = (double*) calloc(num_elements, sizeof(double));
    double *host_R = (double*) calloc(1, sizeof(double));

    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);

    gR.value = am_alloc(sizeof(double) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(double) * num_elements, acc[1], 0);

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
    }    

    am_copy(gX.values, host_X, sizeof(double) * num_elements);
    am_copy(gR.value, host_R, sizeof(double) * 1);

    gR.offValue = 0;
    gX.offValues = 0;

    gX.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseDreduce(&gR, &gX, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i];
    }

    bool ispassed = 1;

    am_copy(host_R, gR.value, sizeof(double) * 1);

    if (host_res[0] != host_R[0])
    {
        ispassed = 0;
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_R);
    am_free(gR.value);
    am_free(gX.values);

    return 0; 
}
