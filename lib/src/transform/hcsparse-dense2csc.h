#include "hcsparse.h"

template <typename T>
hcsparseStatus
dense2csc (hcsparseControl* control,
           int m,
           int n,
           const T *A,
           T *cscValA,
           int *cscColPtrA,
           int *cscRowIndA) {

  hc::accelerator acc = (control->accl_view).get_accelerator();

  hc::extent<1> grdExt(1);
  hc::tiled_extent<1> t_ext = grdExt.tile(1);
  hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
  {
      cscColPtrA[0] = 0;
      int count = 0, col_off = 0;
      for (int i = 0; i < n; i++)
      {
          for (int j = 0; j < m; j++)
          {
              if (A[j*n + i] != 0)
              {
                  cscValA[count] = A[j*n+i];
                  cscRowIndA[count] = j;
                  col_off++;
                  count++;
              }
          }
          cscColPtrA[i+1] = col_off;
      }
  });

  return hcsparseSuccess;
}

