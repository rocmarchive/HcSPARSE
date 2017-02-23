#include <hcsparse.h>
#include <iostream>
#include "hc_am.hpp"
int main()
{
    hcsparseScalar gR;
    hcdenseVector gX;

    std::vector<accelerator>acc = accelerator::get_all();
    accelerator_view accl_view = (acc[1].create_view()); 

    hcsparseControl control(accl_view);

     /* Test New APIs */
    hcsparseHandle_t handle;
    hcsparseStatus_t status1;
    hcsparseMatDescr_t descrA;
    hc::accelerator accl;
    hc::accelerator_view av = accl.get_default_view();

    status1 = hcsparseCreate(&handle, &av);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error Initializing the sparse library."<<std::endl;
      return -1;
    }
    std::cout << "Successfully initialized sparse library"<<std::endl;

    hcsparseDirection_t dir = HCSPARSE_DIRECTION_ROW;

    int m = 64;
    int n = 259;

    status1 = hcsparseCreateMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    std::cout << "successfully created mat descriptor"<<std::endl;

    float *devA = am_alloc(sizeof(float)*m*n, acc[1], 0);
    int lda = m;
    int *nnzPerRowColumn = am_alloc(sizeof(float)*m, acc[1], 0);
    int *nnz = (int *)am_alloc(sizeof(int) * 1, acc[1], 0);

    float *hostA = (float*)calloc (m*n, sizeof(float));
    int *nnzPerRowColumn_h = (int *)calloc(m, sizeof(int));
    int nnz_h;
    int *nnzPerRowColumn_res = (int *)calloc(m, sizeof(int));
    int nnz_res;

    srand (time(NULL));
    for (int i = 0; i < m*n; i++)
    {
        hostA[i] = rand()%100;
    }    

    control.accl_view.copy(hostA, devA, m*n*sizeof(float));

    hcsparseStatus_t stat = hcsparseSnnz(handle, dir, m, n, descrA, devA, lda,
                                         nnzPerRowColumn, nnz);

    control.accl_view.copy(nnzPerRowColumn, nnzPerRowColumn_res, m*sizeof(int));
    control.accl_view.copy(&nnz, &nnz_res, 1*sizeof(int));

    std::cout <<"Sparse nnz done!!" << std::endl;

    for (int i = 0; i < m; i++) {
      int rowCount = 0;
      for (int j = 0; j < n; j++) {
         if ( hostA[i*n+j] != 0) {
           rowCount++;
           nnz_h++;
         }
      }
      nnzPerRowColumn_h[i] = rowCount;
    }

    bool ispassed = 1;
    for (int i = 0; i < m; i++) {
      if (nnzPerRowColumn_res[i] != nnzPerRowColumn_h[i]) {
         ispassed = 0;
         std::cout << "nnPerRowColumn_h[" << i << "] = " << 
                   nnzPerRowColumn_h[i] << " nnzPerRowColumn_res[" 
                   << i << "] = " << nnzPerRowColumn_res[i] << std::endl;
      }
    }
    
    status1 = hcsparseDestroyMatDescr(&descrA);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "error creating mat descrptr"<<std::endl;
      return -1;
    }
    std::cout << "successfully created mat descriptor"<<std::endl;

    status1 = hcsparseDestroy(&handle);
    if (status1 != HCSPARSE_STATUS_SUCCESS) {
      std::cout << "Error DeInitializing the sparse library."<<std::endl;
      return -1;
    }
    std::cout << "Successfully deinitialized sparse library"<<std::endl;
   
    /* End - Test of New APIs */
   return 0; 
}
