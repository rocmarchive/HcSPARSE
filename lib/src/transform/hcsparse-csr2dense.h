#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void
fill_zero (ulong size,
           Concurrency::array_view<T> &values,
           const hcsparseControl* control)
{
    Concurrency::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        int i = tidx.global[0];
        if (i < size)
            values[i] = 0;
    });
}

template <typename T>
hcsparseStatus
csr2dense(const hcsparseCsrMatrix* csr,
          hcdenseMatrix* A,
          const hcsparseControl* control)
{
    ulong dense_size = csr->num_cols * csr->num_rows;
    assert(csr->num_cols == A->num_cols && csr->num_rows == A->num_rows);

    Concurrency::array_view<int> *offsets = static_cast<Concurrency::array_view<int> *>(csr->rowOffsets);
    Concurrency::array_view<int> *indices = static_cast<Concurrency::array_view<int> *>(csr->colIndices);
    Concurrency::array_view<T> *values = static_cast<Concurrency::array_view<T> *>(csr->values);

    Concurrency::array_view<T> *Avalues = static_cast<Concurrency::array_view<T> *>(A->values);

    fill_zero<T> (dense_size, *Avalues, control);

    return transform_csr_2_dense<T> (csr->num_nonzeros, *offsets, *indices, *values,
                                     csr->num_rows, csr->num_cols, *Avalues, control);
}
