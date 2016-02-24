#ifndef _HCSPARSE_PRECOND_UTILS_H_
#define _HCSPARSE_PRECOND_UTILS_H_

#define GROUP_SIZE 256
#define WAVE_SIZE 64

template <typename T, bool inverse>
void extract_diagonal_kernel ( const long num_rows,
                               hc::array_view<T> &diag,
                               const hc::array_view<int> &csr_row_offsets,
                               const hc::array_view<int> &csr_col_indices,
                               const hc::array_view<T> &csr_values,
                               long subwave_size,
                               hcsparseControl *control)
{
    int predicted = subwave_size * num_rows;

    int global_work_size = GROUP_SIZE * ( ( predicted + GROUP_SIZE - 1 ) / GROUP_SIZE );

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
    { 
        const int global_id   = tidx.global[0];         
        const int local_id    = tidx.local[0];          
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size; 
        const int num_vectors = grdExt[0] / subwave_size;
        for(int row = vector_id; row < num_rows; row += num_vectors)
        {
            const int row_start = csr_row_offsets[row];
            const int row_end   = csr_row_offsets[row+1];
            for(int j = row_start + thread_lane; j < row_end; j += subwave_size)
            {
                if (csr_col_indices[j] == row)
                {
                    if (inverse)
                        diag[row] = (T) 1.0 / csr_values[j];
                    else
                         diag[row] = csr_values[j];
                    break;
                }
            }
        }
    });
}

template<typename T, bool inverse = false>
hcsparseStatus
extract_diagonal (hcdenseVector* pDiag,
                  const hcsparseCsrMatrix* pA,
                  hcsparseControl* control)
{
    if (pDiag->values == nullptr)
    {
        return hcsparseInvalid;
    }

    long size = pA->num_rows;

    long nnz_per_row = pA->nnz_per_row();
    long subwave_size = WAVE_SIZE;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(WAVE_SIZE > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    hc::array_view<T> *av_Diag = static_cast<hc::array_view<T> *>(pDiag->values);
    hc::array_view<T> *av_Amat = static_cast<hc::array_view<T> *>(pA->values); 
    hc::array_view<int> *av_ArowOff = static_cast<hc::array_view<int> *>(pA->rowOffsets); 
    hc::array_view<int> *av_AcolInd = static_cast<hc::array_view<int> *>(pA->colIndices); 

    extract_diagonal_kernel<T, inverse> (size, *av_Diag, *av_ArowOff, *av_AcolInd, *av_Amat, subwave_size, control);

    return hcsparseSuccess;
}

#endif
