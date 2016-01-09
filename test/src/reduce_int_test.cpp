#include <hcsparse.h>
#include <iostream>
int main()
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

    bool ispassed = 1;
    array_view<int> *av_res = static_cast<array_view<int> *>(gR.value);
    if (host_res[0] != (*av_res)[0])
    {
        ispassed = 0;
    }

    std::cout << (ispassed?"TEST PASSED":"TEST FAILED") << std::endl;

    hcsparseTeardown();

    return 0; 
}
