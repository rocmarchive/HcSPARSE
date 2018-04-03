#include "hcsparse.h"

#define THREADS 16

template <typename T>
void transpose_kernel (hcsparseControl* control,
                       int rows,
                       int cols,
                       const T* A,
                       T* transA)
{
   hc::extent<2> grdExt((rows + THREADS-1) & ~(THREADS-1), (cols + THREADS-1) & ~(THREADS-1));
   hc::tiled_extent<2> t_ext = grdExt.tile(THREADS, THREADS);
   hc::parallel_for_each(control->accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) [[hc]] {
     int gidx = tidx.global[1];
     int gidy = tidx.global[0];

     if (gidx < cols && gidy < rows) {
       unsigned int index_in = gidy * cols + gidx;
       unsigned int index_out = gidx * rows + gidy;

       transA[index_in] = A[index_out];
     }
   }).wait();
}

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

  ulong dense_size = m * n;
  T* transA = (T*)am_alloc(dense_size * sizeof (T), acc, 0);

  // Transpose the matrix so it will be in row major format
  transpose_kernel<T>(control, m, n, A, transA);

  // Perform dense2csr conversion
  dense2csr(control, n, m, transA, cscValA, cscColPtrA, cscRowIndA);

  am_free(transA);

  return hcsparseSuccess;
}

