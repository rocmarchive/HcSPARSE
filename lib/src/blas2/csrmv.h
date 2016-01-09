#include "hcsparse.h"

#define WAVE_SIZE 64
#define SUBWAVE_SIZE 64
#define WG_SIZE 256
#define INDEX_TYPE uint
#define SIZE_TYPE ulong
#define GLOBAL_SIZE WG_SIZE
#define EXTENDED_PRECISION 0

#ifndef INDEX_TYPE
#error "INDEX_TYPE undefined!"
#endif

#ifndef SIZE_TYPE
#error "SIZE_TYPE undefined!"
#endif

#ifndef WG_SIZE
#error "WG_SIZE undefined!"
#endif

#ifndef WAVE_SIZE
#error "WAVE_SIZE undefined!"
#endif

#ifndef SUBWAVE_SIZE
#error "SUBWAVE_SIZE undefined!"
#endif

#if ( (SUBWAVE_SIZE > WAVE_SIZE) || (SUBWAVE_SIZE != 2 && SUBWAVE_SIZE != 4 && SUBWAVE_SIZE != 8 && SUBWAVE_SIZE != 16 && SUBWAVE_SIZE != 32 && SUBWAVE_SIZE != 64) )
#error "SUBWAVE_SIZE is not  a power of two!"
#endif

// Knuth's Two-Sum algorithm, which allows us to add together two floating
// point numbers and exactly tranform the answer into a sum and a
// rounding error.
// Inputs: x and y, the two inputs to be aded together.
// In/Out: *sumk_err, which is incremented (by reference) -- holds the
//         error value as a result of the 2sum calculation.
// Returns: The non-corrected sum of inputs x and y.
template <typename T>
inline T two_sum( T x,
        T y,
        T &sumk_err) __attribute__((hc, cpu))
{
    const T sumk_s = x + y;
#ifdef EXTENDED_PRECISION
    // We use this 2Sum algorithm to perform a compensated summation,
    // which can reduce the cummulative rounding errors in our SpMV summation.
    // Our compensated sumation is based on the SumK algorithm (with K==2) from
    // Ogita, Rump, and Oishi, "Accurate Sum and Dot Product" in
    // SIAM J. on Scientific Computing 26(6) pp 1955-1988, Jun. 2005.

    // 2Sum can be done in 6 FLOPs without a branch. However, calculating
    // double precision is slower than single precision on every existing GPU.
    // As such, replacing 2Sum with Fast2Sum when using DPFP results in slightly
    // better performance. This is especially true on non-workstation GPUs with
    // low DPFP rates. Fast2Sum is faster even though we must ensure that
    // |a| > |b|. Branch divergence is better than the DPFP slowdown.
    // Thus, for DPFP, our compensated summation algorithm is actually described
    // by both Pichat and Neumaier in "Correction d'une somme en arithmetique
    // a virgule flottante" (J. Numerische Mathematik 19(5) pp. 400-406, 1972)
    // and "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
    // Summen (ZAMM Z. Angewandte Mathematik und Mechanik 54(1) pp. 39-51,
    // 1974), respectively.
    if (hc::fast_math::fabs(x) < hc::fast_math::fabs(y))
    {
        const T swap = x;
        x = y;
        y = swap;
    }
    sumk_err += (y - (sumk_s - x));
    // Original 6 FLOP 2Sum algorithm.
    //T bp = sumk_s - x;
    //(*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
#endif
    return sumk_s;
}

// Performs (x_vals * x_vec) + y using an FMA.
// Ideally, we would perform an error-free transformation here and return the
// appropriate error. However, the EFT of an FMA is very expensive. As such,
// if we are in EXTENDED_PRECISION mode, this function devolves into two_sum
// with x_vals and x_vec inputs multiplied separately from the compensated add.
template <typename T>
inline T two_fma( const T x_vals,
        const T x_vec,
        T y,
        T &sumk_err ) __attribute__((hc, cpu))
{
#ifdef EXTENDED_PRECISION
    T x = x_vals * x_vec;
    const T sumk_s = x + y;
    if (hc::fast_math::fabs(x) < hc::fast_math::fabs(y))
    {
        const T swap = x;
        x = y;
        y = swap;
    }
    sumk_err += (y - (sumk_s - x));
    // 2Sum in the FMA case. Poor performance on low-DPFP GPUs.
    //const T bp = fma(-x_vals, x_vec, sumk_s);
    //(*sumk_err) += (fma(x_vals, x_vec, -(sumk_s - bp)) + (y - bp));
    return sumk_s;
#else
    return fma(x_vals, x_vec, y);
#endif
}

// A method of doing the final reduction without having to copy and paste
// it a bunch of times.
// The EXTENDED_PRECISION section is done as part of the PSum2 addition,
// where we take temporary sums and errors for multiple threads and combine
// them together using the same 2Sum method.
// Inputs:  cur_sum: the input from which our sum starts
//          err: the current running cascade error for this final summation
//          partial: the local memory which holds the values to sum
//                  (we eventually use it to pass down temp. err vals as well)
//          lid: local ID of the work item calling this function.
//          thread_lane: The lane within this SUBWAVE for reduction.
//          round: This parallel summation method operates in multiple rounds
//                  to do a parallel reduction. See the blow comment for usage.
template <typename T>
inline T sum2_reduce( T cur_sum,
        T &err,
        T *partial,
        const INDEX_TYPE lid,
        const INDEX_TYPE thread_lane,
        const INDEX_TYPE round,
        hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
{
    if (SUBWAVE_SIZE > round)
    {
#ifdef EXTENDED_PRECISION
        const unsigned int partial_dest = lid + round;
        if (thread_lane < round)
            cur_sum  = two_sum(cur_sum, partial[partial_dest], err);
        // We reuse the LDS entries to move the error values down into lower
        // threads. This saves LDS space, allowing higher occupancy, but requires
        // more barriers, which can reduce performance.
        tidx.barrier.wait();
        // Have all of those upper threads pass their temporary errors
        // into a location that the lower threads can read.
        if (thread_lane >= round)
            partial[lid] = err;
        tidx.barrier.wait();
        if (thread_lane < round) { // Add those errors in.
            err += partial[partial_dest];
            partial[lid] = cur_sum;
        }
#else
        // This is the more traditional reduction algorithm. It is up to
        // 25% faster (about 10% on average -- potentially worse on devices
        // with low double-precision calculation rates), but can result in
        // numerical inaccuracies, especially in single precision.
        cur_sum += partial[lid + round];
        tidx.barrier.wait();
        partial[lid] = cur_sum;
#endif
    }
    return cur_sum;
}

// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// T - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row
template <typename T>
void csrmv_vector_kernel (const INDEX_TYPE num_rows,
                          const hc::array_view<T, 1> &alpha,
                          const SIZE_TYPE off_alpha,
                          const hc::array_view<INDEX_TYPE, 1> &row_offset,
                          const hc::array_view<INDEX_TYPE, 1> &col,
                          const hc::array_view<T, 1> &val,
                          const hc::array_view<T, 1> &x,
                          const SIZE_TYPE off_x,
                          const hc::array_view<T, 1> &beta,
                          const SIZE_TYPE off_beta,
                          hc::array_view<T, 1> &y,
                          const SIZE_TYPE off_y,
                          const hcsparseControl *control)
{
    hc::extent<1> grdExt( WG_SIZE );
    hc::tiled_extent<1> t_ext = grdExt.tile(WG_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [&] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
    {
        tile_static T sdata [WG_SIZE + SUBWAVE_SIZE / 2];

        //const int vectors_per_block = WG_SIZE/SUBWAVE_SIZE;
        const INDEX_TYPE global_id   = tidx.global[0];         // global workitem id
        const INDEX_TYPE local_id    = tidx.local[0];          // local workitem id
        const INDEX_TYPE thread_lane = local_id & (SUBWAVE_SIZE - 1);
        const INDEX_TYPE vector_id   = global_id / SUBWAVE_SIZE; // global vector id
        const INDEX_TYPE num_vectors = t_ext[0] / SUBWAVE_SIZE;

        const T _alpha = alpha[off_alpha];
        const T _beta = beta[off_beta];

        for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
        {
            const INDEX_TYPE row_start = row_offset[row];
            const INDEX_TYPE row_end   = row_offset[row+1];
            T sum = 0.;

            T sumk_e = 0.;
            // It is about 5% faster to always multiply by alpha, rather than to
            // check whether alpha is 0, 1, or other and do different code paths.
            for(INDEX_TYPE j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
                sum = two_fma<T> (_alpha * val[j], x[off_x + col[j]], sum, sumk_e);
            T new_error = 0.;
            sum = two_sum<T> (sum, sumk_e, new_error);

            // Parallel reduction in shared memory.
           sdata[local_id] = sum;

           // This compensated summation reduces cummulative rounding errors,
           // which can become a problem on GPUs because our reduction order is
           // different than what would be used on a CPU.
           // It is based on the PSumK algorithm (with K==2) from
           // Yamanaka, Ogita, Rump, and Oishi, "A Parallel Algorithm of
           // Accurate Dot Product," in the Journal of Parallel Computing,
           // 34(6-8), pp. 392-410, Jul. 2008.
           #pragma unroll
           for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
           {
               tidx.barrier.wait();
               sum = sum2_reduce<T> (sum, new_error, sdata, local_id, thread_lane, i, tidx);
           }

           if (thread_lane == 0)
           {
               if (_beta == 0)
                    y[off_y + row] = sum + new_error;
               else
               {
                   sum = two_fma<T> (_beta, y[off_y + row], sum, new_error);
                   y[off_y + row] = sum + new_error;
               }
           }
       }
    });
}

template<typename T>
hcsparseStatus
csrmv_vector(const hcsparseScalar* pAlpha,
             const hcsparseCsrMatrix* pMatx,
             const hcdenseVector* pX,
             const hcsparseScalar* pBeta,
             hcdenseVector* pY,
             const hcsparseControl *control)
{
    uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
#if WAVE_SIZE > 32
{
    //this apply only for devices with wavefront > 32 like AMD(64)
    if (nnz_per_row < 64) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 32
}
}
#endif
    if (nnz_per_row < 32) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 16
}
    if (nnz_per_row < 16) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 8
}
    if (nnz_per_row < 8)  {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 4
}
    if (nnz_per_row < 4)  {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 2
}

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    uint predicted = SUBWAVE_SIZE * pMatx->num_rows;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of WG_SIZE. Don't know if that
    // have an impact on performance
    uint global_work_size =
            WG_SIZE * ((predicted + WG_SIZE - 1 ) / WG_SIZE);

    if( global_work_size < WG_SIZE)
    {
#define GLOBAL_SIZE WG_SIZE
    }

    hc::array_view<T> *avAlpha = static_cast<hc::array_view<T> *>(pAlpha->value);
    hc::array_view<INDEX_TYPE> *avMatx_rowOffsets = static_cast<hc::array_view<INDEX_TYPE> *>(pMatx->rowOffsets);
    hc::array_view<INDEX_TYPE> *avMatx_colIndices = static_cast<hc::array_view<INDEX_TYPE> *>(pMatx->colIndices);
    hc::array_view<T> *avMatx_values = static_cast<hc::array_view<T> *>(pMatx->values);
    hc::array_view<T> *avX_values = static_cast<hc::array_view<T> *>(pX->values);
    hc::array_view<T> *avBeta = static_cast<hc::array_view<T> *>(pBeta->value);
    hc::array_view<T> *avY_values = static_cast<hc::array_view<T> *>(pY->values);

    csrmv_vector_kernel<T> (pMatx->num_rows, *avAlpha, pAlpha->offset(),
                         *avMatx_rowOffsets, *avMatx_colIndices, *avMatx_values,
                         *avX_values, pX->offset(), *avBeta,
                         pBeta->offset(), *avY_values, pY->offset(), control);

    return hcsparseSuccess;
}

template <typename T>
void
csrmv_adaptive_kernel(const hc::array_view<T, 1> &vals,
                      const hc::array_view<INDEX_TYPE, 1> &cols,
                      const hc::array_view<INDEX_TYPE, 1> &rowPtrs,
                      const hc::array_view<T, 1> &vec,
                      hc::array_view<T, 1> &out,
                      const hc::array_view<INDEX_TYPE, 1> &rowBlocks,
                      const hc::array_view<T, 1> &pAlpha,
                      const hc::array_view<T, 1> &pBeta,
                      const hcsparseControl *control)
{
}

template <typename T>
hcsparseStatus
csrmv_adaptive( const hcsparseScalar* pAlpha,
                const hcsparseCsrMatrix* pCsrMatx,
                const hcdenseVector* pX,
                const hcsparseScalar* pBeta,
                hcdenseVector* pY,
                const hcsparseControl *control )
{
    //if(control->extended_precision)
    {
//#define EXTENDED_PRECISION 1
    }

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of WG_SIZE. Don't know if that
    // have an impact on performance
    // Setting global work size to half the row block size because we are only
    // using half the row blocks buffer for actual work.
    // The other half is used for the extended precision reduction.
    uint global_work_size = ( (pCsrMatx->rowBlockSize/2) - 1 ) * WG_SIZE;

    if( global_work_size < WG_SIZE)
    {
#define GLOBAL_SIZE WG_SIZE
    }

    hc::array_view<T> *avCsrMatx_values = static_cast<hc::array_view<T> *>(pCsrMatx->values);
    hc::array_view<INDEX_TYPE> *avColIndices = static_cast<hc::array_view<INDEX_TYPE> *>(pCsrMatx->colIndices);
    hc::array_view<INDEX_TYPE> *avRowOffsets = static_cast<hc::array_view<INDEX_TYPE> *>(pCsrMatx->rowOffsets);
    hc::array_view<T> *avX_values = static_cast<hc::array_view<T> *>(pX->values);
    hc::array_view<T> *avY_values = static_cast<hc::array_view<T> *>(pY->values);
    hc::array_view<INDEX_TYPE> *avRowBlocks = static_cast<hc::array_view<INDEX_TYPE> *>(pCsrMatx->rowBlocks);
    hc::array_view<T> *avAlpha = static_cast<hc::array_view<T> *>(pAlpha->value);
    hc::array_view<T> *avBeta = static_cast<hc::array_view<T> *>(pBeta->value);

    csrmv_adaptive_kernel<T> (*avCsrMatx_values, *avColIndices,
                           *avRowOffsets, *avX_values,
                           *avY_values, *avRowBlocks,
                           *avAlpha ,*avBeta, control);

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv (const hcsparseScalar *pAlpha,
       const hcsparseCsrMatrix *pCsrMatx,
       const hcdenseVector *pX,
       const hcsparseScalar *pBeta,
       hcdenseVector *pY,
       const hcsparseControl *control)
{
    if( (pCsrMatx->rowBlocks == nullptr) && (pCsrMatx->rowBlockSize == 0) )
    {
        // Call Vector CSR Kernels
        return csrmv_vector<T>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
    }
    else
    {
        if( ( pCsrMatx->rowBlocks == nullptr ) || ( pCsrMatx->rowBlockSize == 0 ) )
        {
            // rowBlockSize varible is not zero but no pointer
            return hcsparseInvalid;
        }

        // Call adaptive CSR kernels
        return csrmv_adaptive<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );

    }
}
