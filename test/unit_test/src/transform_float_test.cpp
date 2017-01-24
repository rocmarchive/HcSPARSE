#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main()
{
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    float *host_R = (float*) calloc(num_elements, sizeof(float));
    float *host_res = (float*) calloc(num_elements, sizeof(float));
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_Y = (float*) calloc(num_elements, sizeof(float));

    hcsparseSetup();
    hcsparseInitVector(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gX.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gY.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);
    gR.values = am_alloc(sizeof(float) * num_elements, acc[1], 0);

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    control.accl_view.copy(host_X, gX.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_Y, gY.values, sizeof(float) * num_elements);
    control.accl_view.copy(host_R, gR.values, sizeof(float) * num_elements);

    gR.offValues = 0;
    gX.offValues = 0;
    gY.offValues = 0;

    gR.num_values = num_elements;
    gX.num_values = num_elements;
    gY.num_values = num_elements;

    hcsparseStatus status;

    bool ispassed = 1;

    for (int j = 0; j < 4; j++)
    {
        switch(j)
        {
            case 0:
                status = hcdenseSadd(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] + host_Y[i];
                }
                break;
            case 1:
                status = hcdenseSsub(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] - host_Y[i];
                }
                break;
            case 2:
                status = hcdenseSmul(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] * host_Y[i];
                }
                break;
            case 3:
                status = hcdenseSdiv(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] / host_Y[i];
                }
                break;
        }

        control.accl_view.copy(gR.values, host_R, sizeof(float) * num_elements);

        for (int i = 0; i < num_elements; i++)
        {
            if (host_res[i] != host_R[i])
            {
                switch(j)
                {
                    case 0:
                        std::cout << "ADD TEST FAILED" << std::endl;
                        break;
                    case 1:
                        std::cout << "SUB TEST FAILED" << std::endl;
                        break;
                    case 2:
                        std::cout << "MUL TEST FAILED" << std::endl;
                        break;
                    case 3:
                        std::cout << "DIV TEST FAILED" << std::endl;
                        break;
                }
                ispassed = 0;
                break;
            }
        }
    }

    if (ispassed)
        std::cout << "TEST PASSED" << std::endl;

    hcsparseTeardown();

    free(host_R);
    free(host_res);
    free(host_X);
    free(host_Y);
    am_free(gR.values);
    am_free(gX.values);
    am_free(gY.values);

    return 0; 
}
