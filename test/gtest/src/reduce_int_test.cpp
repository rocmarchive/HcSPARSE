#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(reduce_int_test, func_check)
{
    hcsparseScalar gR;
    hcdenseVector gX;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);

    int num_elements = 100;
    gR.value = am_alloc(sizeof(int) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(int) * num_elements, acc[1], 0);

    int *host_res = (int*) calloc(1, sizeof(int));
    int *host_X = (int*) calloc(num_elements, sizeof(int));
    int *host_R = (int*) calloc(1, sizeof(int));

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

    control.accl_view.copy(gR.value, host_R, sizeof(int) * 1);

    int diff = std::abs(host_res[0] - host_R[0]);
    EXPECT_LT(diff, TOLERANCE);

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_R);
    am_free(gR.value);
    am_free(gX.values);
}
