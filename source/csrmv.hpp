#include "hcsparse.h"

#define WAVE_SIZE 64
#define SUBWAVE_SIZE 64
#define WG_SIZE 256
#define INDEX_TYPE uint
#define SIZE_TYPE ulong
#define VALUE_TYPE float
#define GLOBAL_SIZE WG_SIZE
#define EXTENDED_PRECISION 1
#define ROWS_FOR_VECTOR 1
#define BLOCK_MULTIPLIER 3
#define BLOCKSIZE 1024
#define WGBITS 24
#define ROWBITS 32

#ifndef INDEX_TYPE
#error "INDEX_TYPE undefined!"
#endif

#ifndef VALUE_TYPE
#error "VALUE_TYPE undefined!"
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
inline VALUE_TYPE two_sum( VALUE_TYPE x,
        VALUE_TYPE y,
        VALUE_TYPE &sumk_err) restrict(amp)
{
    const VALUE_TYPE sumk_s = x + y;
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
    if (fabs(x) < fabs(y))
    {
        const VALUE_TYPE swap = x;
        x = y;
        y = swap;
    }
    sumk_err += (y - (sumk_s - x));
    // Original 6 FLOP 2Sum algorithm.
    //VALUE_TYPE bp = sumk_s - x;
    //(*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
#endif
    return sumk_s;
}

// Performs (x_vals * x_vec) + y using an FMA.
// Ideally, we would perform an error-free transformation here and return the
// appropriate error. However, the EFT of an FMA is very expensive. As such,
// if we are in EXTENDED_PRECISION mode, this function devolves into two_sum
// with x_vals and x_vec inputs multiplied separately from the compensated add.
inline VALUE_TYPE two_fma( const VALUE_TYPE x_vals,
        const VALUE_TYPE x_vec,
        VALUE_TYPE y,
        VALUE_TYPE &sumk_err ) restrict(amp)
{
#ifdef EXTENDED_PRECISION
    VALUE_TYPE x = x_vals * x_vec;
    const VALUE_TYPE sumk_s = x + y;
    if (fabs(x) < fabs(y))
    {
        const VALUE_TYPE swap = x;
        x = y;
        y = swap;
    }
    sumk_err += (y - (sumk_s - x));
    // 2Sum in the FMA case. Poor performance on low-DPFP GPUs.
    //const VALUE_TYPE bp = fma(-x_vals, x_vec, sumk_s);
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
inline VALUE_TYPE sum2_reduce( VALUE_TYPE cur_sum,
        VALUE_TYPE &err,
        VALUE_TYPE *partial,
        const INDEX_TYPE lid,
        const INDEX_TYPE thread_lane,
        const INDEX_TYPE round,
        Concurrency::tiled_index<WG_SIZE> tidx) restrict(amp)
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
// VALUE_TYPE - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row
void csrmv_vector_kernel (const INDEX_TYPE num_rows,
                          const Concurrency::array_view<VALUE_TYPE, 1> &alpha,
                          const SIZE_TYPE off_alpha,
                          const Concurrency::array_view<INDEX_TYPE, 1> &row_offset,
                          const Concurrency::array_view<INDEX_TYPE, 1> &col,
                          const Concurrency::array_view<VALUE_TYPE, 1> &val,
                          const Concurrency::array_view<VALUE_TYPE, 1> &x,
                          const SIZE_TYPE off_x,
                          const Concurrency::array_view<VALUE_TYPE, 1> &beta,
                          const SIZE_TYPE off_beta,
                          Concurrency::array_view<VALUE_TYPE, 1> &y,
                          const SIZE_TYPE off_y)
{
}

template<typename T>
hcsparseStatus
csrmv_vector(const hcsparseScalar* pAlpha,
             const hcsparseCsrMatrix* pMatx,
             const hcdenseVector* pX,
             const hcsparseScalar* pBeta,
             hcdenseVector* pY,
             hcsparseControl control)
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

    if(typeid(T) == typeid(double))
    {
#undef VALUE_TYPE
#define VALUE_TYPE double
    }
    else if(typeid(T) == typeid(float))
    {
#undef VALUE_TYPE
#define VALUE_TYPE float
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

    csrmv_vector_kernel (pMatx->num_rows, *(pAlpha->value), pAlpha->offset(),
                         *(pMatx->rowOffsets), *(pMatx->colIndices), *(pMatx->values),
                         *(pX->values), pX->offset(), *(pBeta->value),
                         pBeta->offset(), *(pY->values), pY->offset());

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv_adaptive( const hcsparseScalar* pAlpha,
                const hcsparseCsrMatrix* pCsrMatx,
                const hcdenseVector* pX,
                const hcsparseScalar* pBeta,
                hcdenseVector* pY,
                hcsparseControl control )
{
    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv (const hcsparseScalar *pAlpha,
       const hcsparseCsrMatrix *pCsrMatx,
       const hcdenseVector *pX,
       const hcsparseScalar *pBeta,
       hcdenseVector *pY,
       hcsparseControl control)
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
