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
    int *host_res = (int*) calloc(1, sizeof(int));
    int *host_X = (int*) calloc(num_elements, sizeof(int));
    int *host_R = (int*) calloc(1, sizeof(int));

    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);

    gR.value = am_alloc(sizeof(int) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(int) * num_elements, acc[1], 0);

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
    }    

    control.accl_view.copy(host_X, gX.values, sizeof(int) * num_elements);
    control.accl_view.copy(host_R, gR.value, sizeof(int) * 1);

    gR.offValue = 0;
    gX.offValues = 0;

    gX.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseIreduce(&gR, &gX, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i];
    }

    bool ispassed = 1;

    control.accl_view.copy(gR.value, host_R, sizeof(int) * 1);

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
