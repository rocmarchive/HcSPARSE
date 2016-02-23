#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(reduce_int_test, func_check)
{
    hcsparseScalar gR;
    hcdenseVector gX;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    int *host_res = (int*) calloc(1, sizeof(int));
    int *host_X = (int*) calloc(num_elements, sizeof(int));
    int *host_R = (int*) calloc(1, sizeof(int));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
    }
    
    array_view<int> dev_X(num_elements, host_X);
    array_view<int> dev_R(1, host_R);

    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);

    gR.value = &dev_R;
    gX.values = &dev_X;

    gR.offValue = 0;
    gX.offValues = 0;

    gX.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseIreduce(&gR, &gX, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i];
    }

    array_view<int> *av_res = static_cast<array_view<int> *>(gR.value);

    int diff = std::abs(host_res[0] - (*av_res)[0]);
    EXPECT_LT(diff, TOLERANCE);

    dev_X.synchronize();
    dev_R.synchronize();

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_R);
}
