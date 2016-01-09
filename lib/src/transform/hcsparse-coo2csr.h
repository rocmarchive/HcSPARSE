#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void coo2csr_kernel ( const hc::array_view<int> &coo_col,
                      const hc::array_view<T> &coo_values,
                      hc::array_view<int> &csr_col,
                      hc::array_view<T> &csr_values,
                      int size,
                      const hcsparseControl* control)
{

    hc::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
    {
        int i = tidx.global[0];
        if (i < size)
        {
            (csr_col)[i] = (coo_col)[i];
            (csr_values)[i] = (coo_values)[i];
        }
    });
}

template <typename T>
hcsparseStatus
coo2csr (const hcsparseCooMatrix* coo,
         hcsparseCsrMatrix* csr,
         const hcsparseControl* control)
{
    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;

    hc::array_view<int> *coo_rowIndices = static_cast<hc::array_view<int> *>(coo->rowIndices);
    hc::array_view<int> *coo_colIndices = static_cast<hc::array_view<int> *>(coo->colIndices);
    hc::array_view<T> *coo_values = static_cast<hc::array_view<T> *>(coo->values);

    hc::array_view<int> *csr_rowOffsets = static_cast<hc::array_view<int> *>(csr->rowOffsets);
    hc::array_view<int> *csr_colIndices = static_cast<hc::array_view<int> *>(csr->colIndices);
    hc::array_view<T> *csr_values = static_cast<hc::array_view<T> *>(csr->values);

    int size = coo->num_nonzeros;
 
    coo2csr_kernel<T> (*coo_colIndices, *coo_values, *csr_colIndices, *csr_values, size, control);

    return indices_to_offsets<int> (size, *csr_rowOffsets, *coo_rowIndices, control); 
}
