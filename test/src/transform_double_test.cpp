#include <hcsparse.h>
#include <iostream>
int main()
{
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

    srand (time(NULL));
    for (int i = 0; i < num_elements; i++)
    {
        host_R[i] = rand()%100;
        host_X[i] = rand()%100;
        host_Y[i] = rand()%100;
    }
    
    array_view<double> dev_R(num_elements, host_R);
    array_view<double> dev_X(num_elements, host_X);
    array_view<double> dev_Y(num_elements, host_Y);

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
                status = hcdenseDadd(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] + host_Y[i];
                }
                break;
            case 1:
                status = hcdenseDsub(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] - host_Y[i];
                }
                break;
            case 2:
                status = hcdenseDmul(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] * host_Y[i];
                }
                break;
            case 3:
                status = hcdenseDdiv(&gR, &gX, &gY, &control);

                for (int i = 0; i < num_elements; i++)
                {
                    host_res[i] = host_X[i] / host_Y[i];
                }
                break;
        }

        array_view<double> *av_res = static_cast<array_view<double> *>(gR.values);
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
