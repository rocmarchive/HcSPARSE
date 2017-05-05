#include <hcsparse.h>

#include <hc_am.hpp>
#include "gtest/gtest.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

#define TOLERANCE 0.01

TEST(nrm2_float_test, func_check)
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
    float *host_res = (float*) calloc(1, sizeof(float));
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_R = (float*) calloc(1, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
    }
    
    gR.value = am_alloc(sizeof(float) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    
    control.accl_view.copy(host_X, gX.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_R, gR.value, sizeof(float) * 1);

    gR.offValue = 0;
    gX.offValues = 0;

    gX.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseSnrm2(&gR, &gX, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += (host_X[i] * host_X[i]);
    }
    host_res[0] = std::sqrt(host_res[0]);

    control.accl_view.copy(gR.value, host_R, sizeof(float) * 1);

    float diff = std::abs(host_res[0] - host_R[0]);
    EXPECT_LT(diff, TOLERANCE);

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_R);
    am_free(gR.value);
    am_free(gX.values);
}
