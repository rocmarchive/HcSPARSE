#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void
fill_zero (int size,
           hc::array_view<T> &values,
           const hcsparseControl* control)
{
    hc::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        int i = tidx.global[0];
        if (i < size)
            values[i] = 0;
    }).wait();
}

template <typename T>
hcsparseStatus
csr2dense(const hcsparseCsrMatrix* csr,
          hcdenseMatrix* A,
          const hcsparseControl* control)
{
    int dense_size = csr->num_cols * csr->num_rows;

    assert(csr->num_cols == A->num_cols && csr->num_rows == A->num_rows);

    hc::array_view<int> *offsets = static_cast<hc::array_view<int> *>(csr->rowOffsets);
    hc::array_view<int> *indices = static_cast<hc::array_view<int> *>(csr->colIndices);
    hc::array_view<T> *values = static_cast<hc::array_view<T> *>(csr->values);

    hc::array_view<T> *Avalues = static_cast<hc::array_view<T> *>(A->values);

    fill_zero<T> (dense_size, *Avalues, control);

    return transform_csr_2_dense<T> (dense_size, *offsets, *indices, *values,
                                     csr->num_rows, csr->num_cols, *Avalues, control);
}

