#include <hcsparse.h>
#include <iostream>
#include <hc_am.hpp>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(dot_float_test, func_check)
{
    hcsparseScalar gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    int num_elements = 1000;
    float *host_res = (float*) calloc(1, sizeof(float));
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_Y = (float*) calloc(num_elements, sizeof(float));
    float *host_R = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    gR.value = am_alloc(sizeof(float) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gY.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);

    control.accl_view.copy(host_R, gR.value, sizeof(float) * 1);
    control.accl_view.copy(host_X, gX.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_Y, gY.values, sizeof(float) * num_elements);

    gR.offValue = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseSdot(&gR, &gX, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i] * host_Y[i];
    }

    control.accl_view.copy(gR.value, host_R, sizeof(float) * 1);

    float diff = std::abs(host_res[0] - host_R[0]);     
    EXPECT_LT(diff, TOLERANCE);

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_Y);
    free(host_R);
    am_free(gR.value);
    am_free(gX.values);
    am_free(gY.values);
}
