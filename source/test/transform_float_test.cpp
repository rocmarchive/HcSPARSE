#include <hcsparse.h>
#include <iostream>
int main()
{
    hcdenseVector gR;
    hcdenseVector gX;
    hcdenseVector gY;

    std::vector<Concurrency::accelerator>acc = Concurrency::accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

    int num_elements = 100;
    float *host_R = (float*) calloc(num_elements, sizeof(float));
    float *host_res = (float*) calloc(num_elements, sizeof(float));
    float *host_X = (float*) calloc(num_elements, sizeof(float));
    float *host_Y = (float*) calloc(num_elements, sizeof(float));

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    Concurrency::array_view<float> dev_R(num_elements, host_R);
    Concurrency::array_view<float> dev_X(num_elements, host_X);
    Concurrency::array_view<float> dev_Y(num_elements, host_Y);

    hcsparseSetup();
    hcsparseInitVector(&gR);
    hcsparseInitVector(&gX);
    hcsparseInitVector(&gY);

    gR.values = &dev_R;
    gX.values = &dev_X;
    gY.values = &dev_Y;

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

        Concurrency::array_view<float> *av_res = static_cast<Concurrency::array_view<float> *>(gR.values);
        for (int i = 0; i < num_elements; i++)
        {
            if (host_res[i] != (*av_res)[i])
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

    return 0; 
}
