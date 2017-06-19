#include "hcsparse.h"

template <typename T>
hcsparseStatus
vector_add (int size, T *A, T *B, const T *alpha,
            const T *beta, T *C, hcsparseControl *control) {
   
   int num_threads = 256;
   int num_blocks = (size - 1) / 256 + 1;

   size_t szLocalWorkSize = num_threads;
   size_t szGlobalWorkSize = num_blocks * num_threads;

   hc::extent<1> grdExt(szGlobalWorkSize);
   hc::tiled_extent<1> t_ext = grdExt.tile(szLocalWorkSize);
   hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
   {
     int global_id = tidx.global[0];
     if (global_id < size)
     {
       C[global_id] += alpha[0] * A[global_id] + beta[0] * B[global_id];
     }

   }).wait();

   return hcsparseSuccess;
} 
