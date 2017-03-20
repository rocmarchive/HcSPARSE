#include "hcsparse.h"

template <typename T>
void transpose_kernel (hcsparseControl* control,
                       int rows,
                       int cols,
                       const T* A,
                       T* transA)
{
   hc::extent<2> grdExt((cols + 15) & ~15, (rows + 15) & ~15);
   hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
   hc::parallel_for_each(control->accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
     int gidx = tidx.global[1];
     int gidy = tidx.global[0];

     if (gidx < cols && gidy < rows) {
       unsigned int index_in = gidy * cols + gidx;
       unsigned int index_out = gidx * rows + gidy;

       transA[index_out] = A[index_in];
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

  //calculate nnz
  int *nnz_locations = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

  int num_nonzeros = 0;

  calculate_num_nonzeros<T> (dense_size, transA, nnz_locations, num_nonzeros, control);

  int *coo_indexes = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

  exclusive_scan<int, EW_PLUS>(dense_size, coo_indexes, nnz_locations, control);

  hcsparseCooMatrix coo;
  hcsparseCsrMatrix csr;

  hcsparseInitCooMatrix(&coo);
  hcsparseInitCsrMatrix(&csr);

  coo.offValues = 0;
  coo.offColInd = 0;
  coo.offRowInd = 0;

  coo.num_nonzeros = num_nonzeros;
  coo.num_rows = m;
  coo.num_cols = n;

  coo.colIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
  coo.rowIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
  coo.values = (T*) am_alloc(num_nonzeros * sizeof(T), acc, 0);

  csr.offValues = 0;
  csr.offColInd = 0;
  csr.offRowOff = 0;

  csr.num_nonzeros = num_nonzeros;
  csr.num_rows = n;
  csr.num_cols = m;

  csr.values = static_cast<T*>(cscValA);
  csr.rowOffsets = static_cast<int*>(cscColPtrA);
  csr.colIndices = static_cast<int*>(cscRowIndA);

  dense_to_coo<T> (dense_size, m, static_cast<int*>(coo.rowIndices), static_cast<int*>(coo.colIndices), static_cast<T*>(coo.values), transA, nnz_locations, coo_indexes, control);

  if (typeid(T) == typeid(double))
      hcsparseDcoo2csr(&coo, &csr, control);
  else
      hcsparseScoo2csr(&coo, &csr, control);

  am_free(nnz_locations);
  am_free(coo_indexes);
  am_free(coo.colIndices);
  am_free(coo.rowIndices);
  am_free(coo.values);

  return hcsparseSuccess;

}

