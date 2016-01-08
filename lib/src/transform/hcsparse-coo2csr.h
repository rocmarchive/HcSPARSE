#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void coo2csr_kernel ( const Concurrency::array_view<int> &coo_col,
                      const Concurrency::array_view<T> &coo_values,
                      Concurrency::array_view<int> &csr_col,
                      Concurrency::array_view<T> &csr_values,
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

    Concurrency::array_view<int> *coo_rowIndices = static_cast<Concurrency::array_view<int> *>(coo->rowIndices);
    Concurrency::array_view<int> *coo_colIndices = static_cast<Concurrency::array_view<int> *>(coo->colIndices);
    Concurrency::array_view<T> *coo_values = static_cast<Concurrency::array_view<T> *>(coo->values);

    Concurrency::array_view<int> *csr_rowOffsets = static_cast<Concurrency::array_view<int> *>(csr->rowOffsets);
    Concurrency::array_view<int> *csr_colIndices = static_cast<Concurrency::array_view<int> *>(csr->colIndices);
    Concurrency::array_view<T> *csr_values = static_cast<Concurrency::array_view<T> *>(csr->values);

    int size = coo->num_nonzeros;
    int num_rows = coo->num_rows + 1;
 
    coo2csr_kernel<T> (*coo_colIndices, *coo_values, *csr_colIndices, *csr_values, size, control);

    return indices_to_offsets<int> (num_rows, size, *csr_rowOffsets, *coo_rowIndices, control); 
}
