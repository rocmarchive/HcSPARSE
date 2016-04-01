#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void csr2coo_kernel (int *coo_col,
                     T *coo_values,
                     const int *csr_col,
                     const T *csr_values,
                     int size,
                     const hcsparseControl* control)
{

    hc::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        int i = tidx.global[0];
        if (i < size)
        {
            coo_col[i] = csr_col[i];
            coo_values[i] = csr_values[i];
        }
    }).wait();
}

template <typename T>
hcsparseStatus
csr2coo (const hcsparseCsrMatrix* csr,
         hcsparseCooMatrix* coo,
         const hcsparseControl* control)
{
    coo->num_rows = csr->num_rows;
    coo->num_cols = csr->num_cols;
    coo->num_nonzeros = csr->num_nonzeros;

    int *coo_rowIndices = static_cast<int*>(coo->rowIndices);
    int *coo_colIndices = static_cast<int*>(coo->colIndices);
    T *coo_values = static_cast<T*>(coo->values);

    int *csr_rowOffsets = static_cast<int*>(csr->rowOffsets);
    int *csr_colIndices = static_cast<int*>(csr->colIndices);
    T *csr_values = static_cast<T*>(csr->values);

    int size = csr->num_nonzeros;

    int num_rows = csr->num_rows;

    csr2coo_kernel<T> (coo_colIndices, coo_values, csr_colIndices, csr_values, size, control);

    return offsets_to_indices<int> (num_rows, size, coo_rowIndices, csr_rowOffsets, control);
}

