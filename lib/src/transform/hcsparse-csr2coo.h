#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void csr2coo_kernel ( Concurrency::array_view<int> &coo_col,
                      Concurrency::array_view<T> &coo_values,
                      const Concurrency::array_view<int> &csr_col,
                      const Concurrency::array_view<T> &csr_values,
                      int size,
                      const hcsparseControl* control)
{

    Concurrency::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        int i = tidx.global[0];
        if (i < size)
        {
            coo_col[i] = csr_col[i];
            coo_values[i] = csr_values[i];
        }
    });
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

    Concurrency::array_view<int> *coo_rowIndices = static_cast<Concurrency::array_view<int> *>(coo->rowIndices);
    Concurrency::array_view<int> *coo_colIndices = static_cast<Concurrency::array_view<int> *>(coo->colIndices);
    Concurrency::array_view<T> *coo_values = static_cast<Concurrency::array_view<T> *>(coo->values);

    Concurrency::array_view<int> *csr_rowOffsets = static_cast<Concurrency::array_view<int> *>(csr->rowOffsets);
    Concurrency::array_view<int> *csr_colIndices = static_cast<Concurrency::array_view<int> *>(csr->colIndices);
    Concurrency::array_view<T> *csr_values = static_cast<Concurrency::array_view<T> *>(csr->values);

    int size = csr->num_nonzeros;
 
    int num_rows = csr->num_rows;
 
    csr2coo_kernel<T> (*coo_colIndices, *coo_values, *csr_colIndices, *csr_values, size, control);

    return offsets_to_indices<int> (num_rows, size, *coo_rowIndices, *csr_rowOffsets, control); 
}
