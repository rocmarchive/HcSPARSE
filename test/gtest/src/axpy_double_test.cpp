#include <hcsparse.h>
#include <iostream>
#include "gtest/gtest.h"

#define TOLERANCE 0.01

TEST(axpy_double_test, func_check)
{
    hcsparseScalar gAlpha;
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    double *host_R = (double*) calloc(num_elements, sizeof(double));
    double *host_res = (double*) calloc(num_elements, sizeof(double));
    double *host_X = (double*) calloc(num_elements, sizeof(double));
    double *host_Y = (double*) calloc(num_elements, sizeof(double));
    double *host_alpha = (double*) calloc(1, sizeof(double));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    host_alpha[0] = rand()%100;

    array_view<double> dev_R(num_elements, host_R);
    array_view<double> dev_X(num_elements, host_X);
    array_view<double> dev_Y(num_elements, host_Y);
    array_view<double> dev_alpha(1, host_alpha);

    hcsparseSetup();
    hcsparseInitScalar(&gAlpha);
    hcsparseInitVector(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gAlpha.value = &dev_alpha;
    gR.values = &dev_R;
    gX.values = &dev_X;
    gY.values = &dev_Y;

    gAlpha.offValue = 0;
    gR.offValues = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gR.num_values = num_elements;
    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    status = hcdenseDaxpy(&gR, &gAlpha, &gX, &gY, &control);

    for (int i = 0; i < num_elements; i++)
    {
        host_res[i] = host_alpha[0] * host_X[i] + host_Y[i];
    }

    array_view<double> *av_res = static_cast<array_view<double> *>(gR.values);
    for (int i = 0; i < num_elements; i++)
    {
        double diff = std::abs(host_res[i] - (*av_res)[i]);
        EXPECT_LT(diff, TOLERANCE);
    }

    dev_R.synchronize();
    dev_X.synchronize();
    dev_Y.synchronize();
    dev_alpha.synchronize();

    hcsparseTeardown();

    free(host_R);
    free(host_res);
    free(host_X);
    free(host_Y);
    free(host_alpha);

}
