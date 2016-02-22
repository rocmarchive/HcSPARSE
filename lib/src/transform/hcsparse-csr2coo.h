#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void csr2coo_kernel (hc::array_view<int> &coo_col,
                     hc::array_view<T> &coo_values,
                     const hc::array_view<int> &csr_col,
                     const hc::array_view<T> &csr_values,
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

    hc::array_view<int> *coo_rowIndices = static_cast<hc::array_view<int> *>(coo->rowIndices);
    hc::array_view<int> *coo_colIndices = static_cast<hc::array_view<int> *>(coo->colIndices);
    hc::array_view<T> *coo_values = static_cast<hc::array_view<T> *>(coo->values);

    hc::array_view<int> *csr_rowOffsets = static_cast<hc::array_view<int> *>(csr->rowOffsets);
    hc::array_view<int> *csr_colIndices = static_cast<hc::array_view<int> *>(csr->colIndices);
    hc::array_view<T> *csr_values = static_cast<hc::array_view<T> *>(csr->values);

    int size = csr->num_nonzeros;

    int num_rows = csr->num_rows;

    csr2coo_kernel<T> (*coo_colIndices, *coo_values, *csr_colIndices, *csr_values, size, control);

    return offsets_to_indices<int> (num_rows, size, *coo_rowIndices, *csr_rowOffsets, control);
}

