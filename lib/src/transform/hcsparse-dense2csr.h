#include "hcsparse.h"

template <typename T>
hcsparseStatus
dense2csr (hcsparseControl* control, int m, int n,
           const T *A, T *csrValA,
           int *csrRowPtrA, int *csrColIndA) {

  hc::accelerator acc = (control->accl_view).get_accelerator();

  ulong dense_size = m * n;

  //calculate nnz
  int *nnz_locations = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

  const T *Avalues = static_cast<const T*>(A);

  int num_nonzeros = 0;

  calculate_num_nonzeros<T> (dense_size, Avalues, nnz_locations, num_nonzeros, control);

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
  csr.num_rows = m;
  csr.num_cols = n;

  csr.values = static_cast<T*>(csrValA);
  csr.rowOffsets = static_cast<int*>(csrRowPtrA);
  csr.colIndices = static_cast<int*>(csrColIndA);

  dense_to_coo<T> (dense_size, n, static_cast<int*>(coo.rowIndices), static_cast<int*>(coo.colIndices), static_cast<T*>(coo.values), Avalues, nnz_locations, coo_indexes, control);

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
template <typename T>
hcsparseStatus
dense2csr (const hcdenseMatrix* A,
           hcsparseCsrMatrix* csr,
           hcsparseControl* control)
{
    hc::accelerator acc = (control->accl_view).get_accelerator();

    ulong dense_size = A->num_cols * A->num_rows;

    //calculate nnz
    int *nnz_locations = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

    T *Avalues = static_cast<T*>(A->values);

    int num_nonzeros = 0;

    calculate_num_nonzeros<T> (dense_size, Avalues, nnz_locations, num_nonzeros, control);

    int *coo_indexes = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

    exclusive_scan<int, EW_PLUS>(dense_size, coo_indexes, nnz_locations, control);

    hcsparseCooMatrix coo;

    hcsparseInitCooMatrix(&coo);

    coo.offValues = 0;
    coo.offColInd = 0;
    coo.offRowInd = 0;

    coo.num_nonzeros = num_nonzeros;
    coo.num_rows = A->num_rows;
    coo.num_cols = A->num_cols;

    coo.colIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
    coo.rowIndices = (int*) am_alloc(num_nonzeros * sizeof(int), acc, 0);
    coo.values = (T*) am_alloc(num_nonzeros * sizeof(T), acc, 0);

    dense_to_coo<T> (dense_size, A->num_cols, static_cast<int*>(coo.rowIndices), static_cast<int*>(coo.colIndices), static_cast<T*>(coo.values), Avalues, nnz_locations, coo_indexes, control);

    if (typeid(T) == typeid(double))
        hcsparseDcoo2csr(&coo, csr, control);
    else
        hcsparseScoo2csr(&coo, csr, control);

    am_free(nnz_locations);
    am_free(coo_indexes);
    am_free(coo.colIndices);
    am_free(coo.rowIndices);
    am_free(coo.values);

    return hcsparseSuccess;
}
