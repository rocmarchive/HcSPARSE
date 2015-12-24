#include "hcsparse.h"

#define WAVE_SIZE 64
#define GROUP_SIZE 256

template<typename T>
void
csrmv( const int num_rows,
        const int subwave_size,
        const Concurrency::array_view<T> &alpha,
        const long off_alpha,
        const Concurrency::array_view<int> &row_offset,
        const Concurrency::array_view<int> &col,
        const Concurrency::array_view<T> &val,
        const Concurrency::array_view<T> &x,
        const size_t ldx,
        const long off_x,
        const Concurrency::array_view<T> &beta,
        const long off_beta,
        Concurrency::array_view<T> &y,
        const size_t ldy,
        const long off_y,
        int curr_col,
        const hcsparseControl *control )
{
    int predicted = subwave_size * num_rows;

    int global_work_size = GROUP_SIZE * ( ( predicted + GROUP_SIZE - 1 ) / GROUP_SIZE );

    Concurrency::extent<1> grdExt(global_work_size);
    Concurrency::tiled_extent< GROUP_SIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUP_SIZE> tidx) restrict(amp)
    { 
        tile_static T sdata[GROUP_SIZE + WAVE_SIZE / 2];
        const int global_id = tidx.global[0];
        const int local_id = tidx.local[0];
        const int thread_lane = local_id & ( subwave_size - 1 );
        const int vector_id = global_id / subwave_size;
        const int num_vectors = t_ext[0] / subwave_size;
        const T _alpha = alpha[ off_alpha ];
        const T _beta = beta[ off_beta ];
        for( int row = vector_id; row < num_rows; row += num_vectors )
        {
            const int row_start = row_offset[ row ];
            const int row_end = row_offset[ row + 1 ];
            T sum = (T)0;
            for( int j = row_start + thread_lane; j < row_end; j += subwave_size )
            {
                if( _alpha == 1 )
                    sum = fma( val[ j ], x[ off_x + ( col[ j ] * ldx ) ], sum );
                else if( _alpha == 0 )
                    sum = 0;
                else
                    sum = fma( _alpha * val[ j ], x[ off_x + ( col[ j ] * ldx ) ], sum );
            }

            sdata[ local_id ] = sum;
            tidx.barrier.wait();
            if( subwave_size > 32 ) sdata[ local_id ] = sum += sdata[ local_id + 32 ];
            tidx.barrier.wait();
            if( subwave_size > 16 ) sdata[ local_id ] = sum += sdata[ local_id + 16 ];
            tidx.barrier.wait();
            if( subwave_size > 8 )  sdata[ local_id ] = sum += sdata[ local_id + 8 ];
            tidx.barrier.wait();
            if( subwave_size > 4 )  sdata[ local_id ] = sum += sdata[ local_id + 4 ];
            tidx.barrier.wait();
            if( subwave_size > 2 )  sdata[ local_id ] = sum += sdata[ local_id + 2 ];
            tidx.barrier.wait();
            if( subwave_size > 1 )                    sum += sdata[ local_id + 1 ];

            if( thread_lane == 0 )
            {
                if( _beta == 1 )
                    y[ off_y + ( row * ldy ) ] = sum + y[ off_y + ( row * ldy ) ];
                else if( _beta == 0 )
                    y[ off_y + ( row * ldy ) ] = sum;
                else
                    y[ off_y + ( row * ldy ) ] = sum + _beta * y[ off_y + ( row * ldy ) ];
            }
        }
    });
}

template<typename T>
void csrmv_batched( const int num_rows,
                    const int nnz_per_row,
                    const Concurrency::array_view<T> &alpha,
                    const long off_alpha,
                    const Concurrency::array_view<int> &rowOffsets,
                    const Concurrency::array_view<int> &colInd,
                    const Concurrency::array_view<T> &values,
                    const Concurrency::array_view<T> &denseB,
                    const size_t ldB,
                    const long off_B,
                    const Concurrency::array_view<T> &beta,
                    const long off_beta,
                    Concurrency::array_view<T> &denseC,
                    const size_t num_rows_C,
                    const size_t num_cols_C,
                    const size_t ldC,
                    const long off_C,
                    const hcsparseControl *control )
{
    int subwave_size = WAVE_SIZE;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if( WAVE_SIZE > 32 )
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if( nnz_per_row < 64 ) { subwave_size = 32; }
    }
    if( nnz_per_row < 32 ) { subwave_size = 16; }
    if( nnz_per_row < 16 ) { subwave_size = 8; }
    if( nnz_per_row < 8 )  { subwave_size = 4; }
    if( nnz_per_row < 4 )  { subwave_size = 2; }

    //  The current implementation of csrmm is implemented as a batched csrmv
    //  The loop iterates on the number of columns in the output matrix, and we increment
    //  the global pointers to the dense B and C matrices a column for each iteration.
    for( int curr_col = 0; curr_col < num_cols_C; ++curr_col )
    {
        csrmv<T> (num_rows, subwave_size, alpha, off_alpha, rowOffsets, colInd, values, denseB, ldB, off_B, beta, off_beta, denseC, ldC, off_C, curr_col, control);
    }
}

template<typename T>
hcsparseStatus
csrmm( const hcsparseScalar *pAlpha,
       const hcsparseCsrMatrix *pSparseCsrA,
       const hcdenseMatrix *pDenseB,
       const hcsparseScalar *pBeta,
       hcdenseMatrix *pDenseC,
       const hcsparseControl *control )
{
   int nnz_per_row = pSparseCsrA->nnz_per_row();

   Concurrency::array_view<T> *avCsrA_values = static_cast<Concurrency::array_view<T> *>(pSparseCsrA->values);
   Concurrency::array_view<int> *avCsrA_colIndices = static_cast<Concurrency::array_view<int> *>(pSparseCsrA->colIndices);
   Concurrency::array_view<int> *avCsrA_rowOffsets = static_cast<Concurrency::array_view<int> *>(pSparseCsrA->rowOffsets);
 
   Concurrency::array_view<T> *avDenseB_values = static_cast<Concurrency::array_view<T> *>(pDenseB->values);
   Concurrency::array_view<T> *avDenseC_values = static_cast<Concurrency::array_view<T> *>(pDenseC->values);

   Concurrency::array_view<T> *avAlpha_value = static_cast<Concurrency::array_view<T> *>(pAlpha->value);
   Concurrency::array_view<T> *avBeta_value = static_cast<Concurrency::array_view<T> *>(pBeta->value);

    csrmv_batched<T> (pSparseCsrA->num_rows, nnz_per_row, *avAlpha_value, pAlpha->offValue, *avCsrA_rowOffsets, *avCsrA_colIndices, 
                      *avCsrA_values, *avDenseB_values, pDenseB->lead_dim, pDenseB->offValues, *avBeta_value, pBeta->offValue, 
                      *avDenseC_values, pDenseC->num_rows, pDenseC->num_cols, pDenseC->lead_dim, pDenseC->offValues, control);

    return hcsparseSuccess;
}
