#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(axpby_float_test, func_check)
{
    hcsparseScalar gAlpha;
    hcsparseScalar gBeta;
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);
    hcsparseSetup();
    hcsparseInitScalar(&gAlpha);
    hcsparseInitScalar(&gBeta);
    hcsparseInitVector(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    int num_elements = 100;
    float *host_R = (float*) calloc(num_elements, sizeof(float));
    float *host_res = (float*) calloc(num_elements, sizeof(float));
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

    gR.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gX.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gY.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gAlpha.value = am_alloc(sizeof(float) * 1, acc[1], 0);
    gBeta.value = am_alloc(sizeof(float) * 1, acc[1], 0);

    gAlpha.offValue = 0;
    gBeta.offValue = 0;
    gR.offValues = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gR.num_values = num_elements;
    gX.num_values = num_elements;
    gY.num_values = num_elements;

    control.accl_view.copy(host_R, gR.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_X, gX.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_Y, gY.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_alpha, gAlpha.value, sizeof(float) * 1);
    control.accl_view.copy(host_beta, gBeta.value, sizeof(float) * 1);

    hcsparseStatus status;

    status = hcdenseSaxpby(&gR, &gAlpha, &gX, &gBeta, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[i] = host_alpha[0] * host_X[i] + host_beta[0] * host_Y[i];
    }

    array_view<float> *av_res = static_cast<array_view<float> *>(gR.values);
    for (int i = 0; i < num_elements; i++)
    {
        float diff = std::abs(host_res[i] - (*av_res)[i]);
        EXPECT_LT(diff, TOLERANCE);
    }

    hcsparseTeardown();

    free(host_R);
    free(host_res);
    free(host_X);
    free(host_Y);
    free(host_alpha);
    free(host_beta);
    am_free(gR.values);
    am_free(gX.values);
    am_free(gY.values);
    am_free(gAlpha.value);
    am_free(gBeta.value);

}
