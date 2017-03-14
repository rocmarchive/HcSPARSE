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

  ulong dense_size = m * n;

  //calculate nnz
  int *nnz_locations = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

  int num_nonzeros = 0;

  calculate_num_nonzeros<T> (dense_size, A, nnz_locations, num_nonzeros, control);

  int *coo_indexes = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

  exclusive_scan<int, EW_PLUS>(dense_size, coo_indexes, nnz_locations, control);

  hcsparseCooMatrix coo;
  hcsparseCscMatrix csc;

  hcsparseInitCooMatrix(&coo);
  hcsparseInitCscMatrix(&csc);

  coo.offValues = 0;
  coo.offColInd = 0;
  coo.offRowInd = 0;

  coo.num_nonzeros = num_nonzeros;
  coo.num_rows = m;
  coo.num_cols = n;

  coo.colIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
  coo.rowIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
  coo.values = (T*) am_alloc(num_nonzeros * sizeof(T), acc, 0);

  csc.offValues = 0;
  csc.offRowInd = 0;
  csc.offColOff = 0;

  csc.num_nonzeros = num_nonzeros;
  csc.num_rows = m;
  csc.num_cols = n;

  csc.values = static_cast<T*>(cscValA);
  csc.colOffsets = static_cast<int*>(cscColPtrA);
  csc.rowIndices = static_cast<int*>(cscRowIndA);

  dense_to_coo<T> (dense_size, n, static_cast<int*>(coo.rowIndices), static_cast<int*>(coo.colIndices), static_cast<T*>(coo.values), A, nnz_locations, coo_indexes, control);

#if 0
  if (typeid(T) == typeid(double))
      hcsparseDcoo2csc(&coo, &csc, control);
  else
      hcsparseScoo2csc(&coo, &csc, control);
#endif

  am_free(nnz_locations);
  am_free(coo_indexes);
  am_free(coo.colIndices);
  am_free(coo.rowIndices);
  am_free(coo.values);

  return hcsparseSuccess;

}
