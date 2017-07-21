#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T>
void
fill_zero (ulong size,
           T *values,
           hcsparseControl* control)
{
    hc::extent<1> grdExt(BLOCK_SIZE * ((size - 1)/BLOCK_SIZE + 1));
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        int i = tidx.global[0];
        if (i < size)
            values[i] = 0;
    });
}

template <typename T>
hcsparseStatus
csr2dense (const hcsparseCsrMatrix* csr,
           hcdenseMatrix* A,
           hcsparseControl* control)
{
    ulong dense_size = csr->num_cols * csr->num_rows;

    assert(csr->num_cols == A->num_cols && csr->num_rows == A->num_rows);

    int *offsets = static_cast<int*>(csr->rowOffsets);
    int *indices = static_cast<int*>(csr->colIndices);
    T *values = static_cast<T*>(csr->values);

    T *Avalues = static_cast<T*>(A->values);

    fill_zero<T> (dense_size, Avalues, control);
    control->accl_view.wait();

    return transform_csr_2_dense<T> (dense_size, offsets, indices, values,
                                     csr->num_rows, csr->num_cols, Avalues, control);
}

