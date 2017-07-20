#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void coo2csr_kernel (const int *coo_col,
                     const T *coo_values,
                     int *csr_col,
                     T *csr_values,
                     int size,
                     hcsparseControl* control)
{

    hc::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) [[hc]]
    {
        int i = tidx.global[0];
        if (i < size)
        {
            csr_col[i] = coo_col[i];
            csr_values[i] = coo_values[i];
        }
    }).wait();
}

template <typename T>
hcsparseStatus
coo2csr (const hcsparseCooMatrix* coo,
         hcsparseCsrMatrix* csr,
         hcsparseControl* control)
{
    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;

    int *coo_rowIndices = static_cast<int*>(coo->rowIndices);
    int *coo_colIndices = static_cast<int*>(coo->colIndices);
    T *coo_values = static_cast<T*>(coo->values);

    int *csr_rowOffsets = static_cast<int*>(csr->rowOffsets);
    int *csr_colIndices = static_cast<int*>(csr->colIndices);
    T *csr_values = static_cast<T*>(csr->values);

    int size = coo->num_nonzeros;
    int num_rows = coo->num_rows + 1;
 
    coo2csr_kernel<T> (coo_colIndices, coo_values, csr_colIndices, csr_values, size, control);
    control->accl_view.wait();

    return indices_to_offsets<int> (num_rows, size, csr_rowOffsets, coo_rowIndices, control); 
}
