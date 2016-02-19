#include "hcsparse.h"

template <typename T>
hcsparseStatus
dense2csr(const hcdenseMatrix* A,
          hcsparseCsrMatrix* csr,
          const hcsparseControl* control)
{
    int dense_size = A->num_cols * A->num_rows;

    //calculate nnz
    int* nnz_location_buf = (int*) calloc (dense_size, sizeof(int));
    hc::array_view<int> nnz_locations(dense_size, nnz_location_buf);

    hc::array_view<T> *Avalues = static_cast<hc::array_view<T> *>(A->values);

    int num_nonzeros = 0;

    calculate_num_nonzeros<T> (dense_size, *Avalues, nnz_locations, num_nonzeros, control);

    int* coo_indexes_buf = (int*) calloc (dense_size, sizeof(int));
    hc::array_view<int> coo_indexes(dense_size, coo_indexes_buf);

    exclusive_scan<int, EW_PLUS>(dense_size, coo_indexes, nnz_locations, control);

    hcsparseCooMatrix coo;

    hcsparseInitCooMatrix(&coo);

    coo.offValues = 0;
    coo.offColInd = 0;
    coo.offRowInd = 0;

    coo.num_nonzeros = num_nonzeros;
    coo.num_rows = A->num_rows;
    coo.num_cols = A->num_cols;

    int* colInd_buf = (int*) calloc (num_nonzeros, sizeof(int));
    int* rowInd_buf = (int*) calloc (num_nonzeros, sizeof(int));
    T* val_buf = (T*) calloc (num_nonzeros, sizeof(T));

    hc::array_view<int> colInd(num_nonzeros, colInd_buf);
    hc::array_view<int> rowInd(num_nonzeros, rowInd_buf);
    hc::array_view<T> values(num_nonzeros, val_buf);

    coo.values = &values;
    coo.colIndices = &colInd;
    coo.rowIndices = &rowInd;

    dense_to_coo<T> (dense_size, A->num_cols, rowInd, colInd, values, *Avalues, nnz_locations, coo_indexes, control);

    if (typeid(T) == typeid(double))
        hcsparseDcoo2csr(&coo, csr, control);
    else
        hcsparseScoo2csr(&coo, csr, control);

    colInd.synchronize();
    rowInd.synchronize();
    values.synchronize();

    free(colInd_buf);
    free(rowInd_buf);
    free(val_buf);
    return hcsparseSuccess;
}
