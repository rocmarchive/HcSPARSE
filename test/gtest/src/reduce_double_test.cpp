#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(reduce_double_test, func_check)
{
    hcsparseScalar gR;
    hcdenseVector gX;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitScalar(&gR);
    hcsparseInitVector(&gX);

    int num_elements = 1000;
    double *host_res = (double*) calloc(1, sizeof(double));
    double *host_X = (double*) calloc(num_elements, sizeof(double));
    double *host_R = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_X[i] = rand()%100;
    }
    
    gR.value = am_alloc(sizeof(double) * 1, acc[1], 0);
    gX.values = am_alloc(sizeof(double) * num_elements, acc[1], 0);

    control.accl_view.copy(host_X, gX.values, sizeof(double) * num_elements);
    control.accl_view.copy(host_R, gR.value, sizeof(double) * 1);

    gR.offValue = 0;
    gX.offValues = 0;

    gX.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseDreduce(&gR, &gX, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[0] += host_X[i];
    }

    control.accl_view.copy(gR.value, host_R, sizeof(double) * 1);
 
    double diff = std::abs(host_res[0] - host_R[0]);
    EXPECT_LT(diff, TOLERANCE);

    hcsparseTeardown();

    free(host_res);
    free(host_X);
    free(host_R);
    am_free(gR.value);
    am_free(gX.values);
}
