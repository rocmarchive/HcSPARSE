#include "hcsparse.h"

#define WAVE_SIZE 64
#define SUBWAVE_SIZE 64
#define WG_SIZE 256
#define INDEX_TYPE int
#define SIZE_TYPE ulong
#define GLOBAL_SIZE WG_SIZE
#define EXTENDED_PRECISION 1

#define ROWS_FOR_VECTOR 1
#define BLOCK_MULTIPLIER 3
#define BLOCKSIZE 256
#define WGBITS 24
#define ROWBITS 32

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

inline int clz(const unsigned int val) __attribute__ ((hc, cpu))
{
    unsigned int temp;
    int counter = 0;

    temp = val;
    while (temp != 0)
    {
        counter++;
        temp = temp >> 1;
    }
    return 32 - counter;
}

inline unsigned long hcsparse_atomic_xor(unsigned long *ptr,
                                         const unsigned long xor_val) __attribute__ ((hc, cpu))
{
    return atomic_fetch_xor((unsigned int*)ptr, (unsigned int)xor_val);
}
inline unsigned long hcsparse_atomic_max(unsigned long *ptr,
                                         const unsigned long compare) __attribute__ ((hc, cpu))
{
    return atomic_fetch_max((unsigned int*)ptr, (unsigned int)compare);
}
inline unsigned long hcsparse_atomic_inc(unsigned long *inc_this) __attribute__ ((hc, cpu))
{
    return atomic_fetch_inc((unsigned int*)inc_this);
}
inline unsigned long hcsparse_atomic_cmpxchg(unsigned long *ptr,
                                             const unsigned long compare,
                                             const unsigned long val) __attribute__ ((hc, cpu))
{
    return atomic_compare_exchange((unsigned int*)ptr, (unsigned int*)&compare, val);
}

template <typename T>
T atomic_add_float_extended(T *ptr,
                            const T temp,
                            T *old_sum ) __attribute__ ((hc, cpu))
{
    unsigned long newVal;
    unsigned long prevVal;
    do
    {
        prevVal = (unsigned long)(*ptr);
	newVal = (unsigned long)(temp + *ptr);
    } while (hcsparse_atomic_cmpxchg((unsigned long *)ptr, prevVal, newVal) != prevVal);

    if (old_sum != 0)
        *old_sum = (T)(prevVal);

    return (T)(newVal);
}

template <typename T>
void atomic_add_float(void *ptr, const T temp ) __attribute__ ((hc, cpu))
{
    atomic_add_float_extended<T> ((T*)ptr, temp, 0);
}

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
                   T &sumk_err) [[hc]]
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
    if (hc::precise_math::fabs(x) < hc::precise_math::fabs(y))
    {
        const T swap = x;
        x = y;
        y = swap;
    }
    sumk_err += (y - (sumk_s - x));
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
                  T &sumk_err ) [[hc]]
{
#ifdef EXTENDED_PRECISION
    T x = x_vals * x_vec;
    const T sumk_s = x + y;
    if (hc::fast_math::fabsf(x) < hc::fast_math::fabsf(y))
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
                      const INDEX_TYPE max_size,
                      hc::tiled_index<1> tidx) [[hc]]
{
    if (max_size > round)
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

template <typename T>
T atomic_two_sum_float (T *x_ptr,
                        T y,
                        T *sumk_err ) __attribute__ ((hc, cpu))
{
    // Have to wait until the return from the atomic op to know what X was.
    T sumk_s = 0.;
#ifdef EXTENDED_PRECISION
    T x;
    sumk_s = atomic_add_float_extended(x_ptr, y, &x);
    if (hc::precise_math::fabs(x) < hc::precise_math::fabs(y))
    {
        const T swap = x;
        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
#else
    atomic_add_float(x_ptr, y);
#endif
    return sumk_s;
}

// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// T - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row
template <typename T>
void csrmv_vector_kernel (const INDEX_TYPE num_rows,
                          const T *alpha,
                          const SIZE_TYPE off_alpha,
                          const int *row_offset,
                          const int *col,
                          const T *val,
                          const T *x,
                          const SIZE_TYPE off_x,
                          const T *beta,
                          const SIZE_TYPE off_beta,
                          T *y,
                          const SIZE_TYPE off_y,
                          const uint global_work_size,
                          hcsparseControl *control)
{
    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(WG_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static T sdata [WG_SIZE + SUBWAVE_SIZE / 2];

        //const int vectors_per_block = WG_SIZE/SUBWAVE_SIZE;
        const INDEX_TYPE global_id   = tidx.global[0];         // global workitem id
        const INDEX_TYPE local_id    = tidx.local[0];          // local workitem id
        const INDEX_TYPE thread_lane = local_id & (SUBWAVE_SIZE - 1);
        const INDEX_TYPE vector_id   = global_id / SUBWAVE_SIZE; // global vector id
        const INDEX_TYPE num_vectors = grdExt[0] / SUBWAVE_SIZE;

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
               sum = sum2_reduce<T> (sum, new_error, sdata, local_id, thread_lane, i, SUBWAVE_SIZE, tidx);
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
    }).wait();
}

template<typename T>
hcsparseStatus
csrmv_vector(const hcsparseScalar* pAlpha,
             const hcsparseCsrMatrix* pMatx,
             const hcdenseVector* pX,
             const hcsparseScalar* pBeta,
             hcdenseVector* pY,
             hcsparseControl *control)
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
        global_work_size = WG_SIZE;
    }

    T *avAlpha = static_cast<T*>(pAlpha->value);
    int *avMatx_rowOffsets = static_cast<int*>(pMatx->rowOffsets);
    int *avMatx_colIndices = static_cast<int*>(pMatx->colIndices);
    T *avMatx_values = static_cast<T*>(pMatx->values);
    T *avX_values = static_cast<T*>(pX->values);
    T *avBeta = static_cast<T*>(pBeta->value);
    T *avY_values = static_cast<T*>(pY->values);

    csrmv_vector_kernel<T> (pMatx->num_rows, avAlpha, pAlpha->offset(),
                            avMatx_rowOffsets, avMatx_colIndices, avMatx_values,
                            avX_values, pX->offset(), avBeta,
                            pBeta->offset(), avY_values, pY->offset(), global_work_size, control);

    return hcsparseSuccess;
}

template <typename T>
void
csrmv_adaptive_kernel (const T *vals,
                       const int *cols,
                       const int *rowPtrs,
                       const T *vec,
                       T *out,
                       unsigned long *rowBlocks,
                       const T *pAlpha,
                       const T *pBeta,
                       const uint global_work_size,
                       hcsparseControl *control)
{
    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(WG_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static T partialSums[WG_SIZE];
        const unsigned int gid = tidx.tile[0];
        const unsigned int lid = tidx.local[0];
        const T alpha = pAlpha[0];
        const T beta = pBeta[0];

        // The row blocks buffer holds a packed set of information used to inform each
        // workgroup about how to do its work:
        //
        // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
        // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
        // |------------Row Information------------|--------^|---WG ID within a long row---|
        // |                                       |    flag/|or # reduce threads for short|
        //
        // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
        // row it will be working on. When one workgroup calculates multiple rows, this
        // rowBlock entry and the next one tell it the range of rows to work on.
        // The lower 24 bits are used whenever multiple workgroups calculate a single long
        // row. This tells each workgroup its ID within that row, so it knows which
        // part of the row to operate on.
        // Alternately, on "short" row blocks, the lower bits are used to communicate
        // the number of threads that should be used for the reduction. Pre-calculating
        // this on the CPU-side results in a noticable performance uplift on many matrices.
        // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
        // know when the first workgroup for that row has finished initializing the output
        // value. While this bit is the same as the first workgroup's flag bit, this
        // workgroup will spin-loop.
        unsigned int row = ((rowBlocks[gid] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));
        unsigned int stop_row = ((rowBlocks[gid + 1] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));
        unsigned int num_rows = stop_row - row;

        // Get the "workgroup within this long row" ID out of the bottom bits of the row block.
        unsigned int wg = rowBlocks[gid] & ((1 << WGBITS) - 1);

        // Any workgroup only calculates, at most, BLOCK_MULTIPLIER*BLOCKSIZE items in a row.
        // If there are more items in this row, we assign more workgroups.
        unsigned int vecStart = (wg * (unsigned int)(BLOCK_MULTIPLIER*BLOCKSIZE)) + rowPtrs[row];
        unsigned int vecEnd = (rowPtrs[row + 1] > vecStart + BLOCK_MULTIPLIER*BLOCKSIZE) ? vecStart + BLOCK_MULTIPLIER*BLOCKSIZE : rowPtrs[row + 1];

        // In here because we don't support 64-bit atomics while working on 64-bit data.
        // As such, we can't use CSR-LongRows. Time to do a fixup -- first WG does the
        // entire row with CSR-Vector. Other rows immediately exit.
        if (num_rows == 0 || (num_rows == 1 && wg)) // CSR-LongRows case
        {
           num_rows = ROWS_FOR_VECTOR;
           stop_row = wg ? row : (row + 1);
           wg = 0;
        }

        T temp_sum = 0.;
        T sumk_e = 0.;
        T new_error = 0.;

        // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
        // If this is zero (long rows) or one (final workgroup in a long row, or a single
        // row in a row block), we want to use the CSR-Vector algorithm(s).
        // We have found, through experimentation, that CSR-Vector is generally faster
        // when working on 2 rows, due to its simplicity and better reduction method.
        if (num_rows > ROWS_FOR_VECTOR)
        {
            // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
            // "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format"
            // for a detailed description of CSR-Stream.
            // In a nutshell, the idea is to use all of the threads to stream the matrix
            // values into the local memory in a fast, coalesced manner. After that, the
            // per-row reductions are done out of the local memory, which is designed
            // to handle non-coalsced accesses.

            // The best method for reducing the local memory values depends on the number
            // of rows. The SC'14 paper discusses a "CSR-Scalar" style reduction where
            // each thread reduces its own row. This yields good performance if there
            // are many (relatively short) rows. However, if they are few (relatively
            // long) rows, it's actually better to perform a tree-style reduction where
            // multiple threads team up to reduce the same row.

            // The calculation below tells you how many threads this workgroup can allocate
            // to each row, assuming that every row gets the same number of threads.
            // We want the closest lower (or equal) power-of-2 to this number --
            // that is how many threads can work in each row's reduction using our algorithm.
            // For instance, with workgroup size 256, 2 rows = 128 threads, 3 rows = 64
            // threads, 4 rows = 64 threads, 5 rows = 32 threads, etc.
            //int numThreadsForRed = get_local_size(0) >> ((CHAR_BIT*sizeof(unsigned int))-clz(num_rows-1));
            const unsigned int numThreadsForRed = wg; // Same calculation as above, done on host.

            // Stream all of this row block's matrix values into local memory.
            // Perform the matvec in parallel with this work.
            const unsigned int col = rowPtrs[row] + lid;
            if (gid != (grdExt[0]/tidx.tile_dim[0] - 1))
            {
                for(int i = 0; i < BLOCKSIZE; i += WG_SIZE)
                    partialSums[lid + i] = alpha * vals[col + i] * vec[cols[col + i]];
            }
            else
            {
                // This is required so that we stay in bounds for vals[] and cols[].
                // Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
                // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
                // The values are within a dGPU's page, which is zeroed out on allocation.
                // However, this may change in the future (e.g. with shared virtual memory.)
                // This causes a minor performance loss because this is the last workgroup
                // to be launched, and this loop can't be unrolled.
                const unsigned int max_to_load = rowPtrs[stop_row] - rowPtrs[row];
                for(int i = 0; i < max_to_load; i += WG_SIZE)
                    partialSums[lid + i] = alpha * vals[col + i] * vec[cols[col + i]];
            }
            tidx.barrier.wait();

            if (numThreadsForRed > 1)
            {
                // In this case, we want to have the workgroup perform a tree-style reduction
                // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
                // a row into {numThreadsForRed} locations in local memory.
                // After that, the entire workgroup does a parallel reduction, and each
                // row ends up with an individual answer.

                // {numThreadsForRed} adjacent threads all work on the same row, so their
                // start and end values are the same.
                // numThreadsForRed guaranteed to be a power of two, so the clz code below
                // avoids an integer divide. ~2% perf gain in EXTRA_PRECISION.
                //size_t st = lid/numThreadsForRed;
                unsigned int local_row = row + (lid >> (31 - clz(numThreadsForRed)));          
                const unsigned int local_first_val = rowPtrs[local_row] - rowPtrs[row];
                const unsigned int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
                const unsigned int threadInBlock = lid & (numThreadsForRed - 1);

                // Not all row blocks are full -- they may have an odd number of rows. As such,
                // we need to ensure that adjacent-groups only work on real data for this rowBlock.
                if (local_row < stop_row)
                {
                    // This is dangerous -- will infinite loop if your last value is within
                    // numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
                    // long induction variable here, though.
                    for(unsigned int local_cur_val = local_first_val + threadInBlock;
                                     local_cur_val < local_last_val;
                                     local_cur_val += numThreadsForRed)
                        temp_sum = two_sum<T> (partialSums[local_cur_val], temp_sum, sumk_e);
                }
                tidx.barrier.wait();

                temp_sum = two_sum<T> (temp_sum, sumk_e, new_error);
                partialSums[lid] = temp_sum;

                // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
                // values sitting in the local memory. This means that, roughly, the beginning of
                // LDS is full up to {workgroup size} entries.
                // Now we perform a parallel reduction that sums together the answers for each
                // row in parallel, leaving us an answer in 'temp_sum' for each row.
                for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
                {
                    tidx.barrier.wait();
                    temp_sum = sum2_reduce<T> (temp_sum, new_error, partialSums, lid, threadInBlock, i, numThreadsForRed, tidx);
                }

                if (threadInBlock == 0 && local_row < stop_row)
                {
                    // All of our write-outs check to see if the output vector should first be zeroed.
                    // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
                    // performance improvement.
                    if (beta != 0.)
                        temp_sum = two_fma<T> (beta, out[local_row], temp_sum, new_error);
                    out[local_row] = temp_sum + new_error;
                }
            }
            else
            {
                // In this case, we want to have each thread perform the reduction for a single row.
                // Essentially, this looks like performing CSR-Scalar, except it is computed out of local memory.
                // However, this reduction is also much faster than CSR-Scalar, because local memory
                // is designed for scatter-gather operations.
                // We need a while loop because there may be more rows than threads in the WG.
                unsigned int local_row = row + lid;
                while (local_row < stop_row)
                {
                    int local_first_val = (rowPtrs[local_row] - rowPtrs[row]);
                    int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
                    temp_sum = 0.;
                    sumk_e = 0.;
                    for (int local_cur_val = local_first_val; local_cur_val < local_last_val; local_cur_val++)
                        temp_sum = two_sum<T> (partialSums[local_cur_val], temp_sum, sumk_e);

                    // After you've done the reduction into the temp_sum register,
                    // put that into the output for each row.
                    if (beta != 0.)
                        temp_sum = two_fma<T> (beta, out[local_row], temp_sum, sumk_e);
                    out[local_row] = temp_sum + sumk_e;
                    local_row += WG_SIZE;
                }
            }
        }
        else if (num_rows >= 1 && !wg) // CSR-Vector case.
        {
            // ^^ The above check says that if this workgroup is supposed to work on <= ROWS_VECTOR
            // number of rows then we should do the CSR-Vector algorithm. If we want this row to be
            // done with CSR-LongRows, then all of its workgroups (except the last one) will have the
            // same stop_row and row. The final workgroup in a LongRow will have stop_row and row
            // different, but the internal "wg" number will be non-zero.

            // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
            // numbers of rows), then it needs to iterate until it reaches the stop_row.
            // We don't check <= stop_row because of the potential for unsigned overflow.
            while (row < stop_row)
            {
                // Any workgroup only calculates, at most, BLOCKSIZE items in this row.
                // If there are more items in this row, we use CSR-LongRows.
                temp_sum = 0.;
                sumk_e = 0.;
                new_error = 0.;
                vecStart = rowPtrs[row];
                vecEnd = rowPtrs[row+1];

                // Load in a bunch of partial results into your register space, rather than LDS (no contention)
                // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
                // Using a long induction variable to make sure unsigned int overflow doesn't break things.
                for (long j = vecStart + lid; j < vecEnd; j+=WG_SIZE)
                {
                    const unsigned int col = cols[(unsigned int)j];
                    temp_sum = two_fma<T> (alpha*vals[(unsigned int)j], vec[col], temp_sum, sumk_e);
                }

                temp_sum = two_sum<T> (temp_sum, sumk_e, new_error);
                partialSums[lid] = temp_sum;

                // Reduce partial sums
                for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
                {
                    tidx.barrier.wait();
                    temp_sum = sum2_reduce<T> (temp_sum, new_error, partialSums, lid, lid, i, WG_SIZE, tidx);
                }

                if (lid == 0UL)
                {
                    if (beta != 0.)
                        temp_sum = two_fma<T> (beta, out[row], temp_sum, new_error);
                    out[row] = temp_sum + new_error;
                }
                row++;
            }
        }
        else
        {
            // In CSR-LongRows, we have more than one workgroup calculating this row.
            // The output values for those types of rows are stored using atomic_add, because
            // more than one parallel workgroup's value makes up the final answer.
            // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
            // the values still left in y will be added in using the atomic_add.
            //
            // Our solution is to have the first workgroup in one of these long-rows cases
            // properly initaizlie the output vector. All the other workgroups working on this
            // row will spin-loop until that workgroup finishes its work.

            // First, figure out which workgroup you are in the row. Bottom 24 bits.
            // You can use that to find the global ID for the first workgroup calculating
            // this long row.
            const unsigned int first_wg_in_row = gid - (rowBlocks[gid] & ((1UL << WGBITS) - 1UL));
            const unsigned int compare_value = rowBlocks[gid] & (1UL << WGBITS);

            // Bit 24 in the first workgroup is the flag that everyone waits on.
            if (gid == first_wg_in_row && lid == 0UL)
            {
                // The first workgroup handles the output initialization.
                T out_val = out[row];
                temp_sum = (beta - 1.) * out_val;
#ifdef EXTENDED_PRECISION
                rowBlocks[grdExt[0]/tidx.tile_dim[0] + gid + 1] = 0UL;
#endif
                hcsparse_atomic_xor(&rowBlocks[first_wg_in_row], (1UL << WGBITS)); // Release other workgroups.
            }
            // For every other workgroup, bit 24 holds the value they wait on.
            // If your bit 24 == first_wg's bit 24, you spin loop.
            // The first workgroup will eventually flip this bit, and you can move forward.
            tidx.barrier.wait();
            while(gid != first_wg_in_row &&
                  lid == 0U &&
                  ((hcsparse_atomic_max(&rowBlocks[first_wg_in_row], 0UL) & (1UL << WGBITS)) == compare_value));
            tidx.barrier.wait();

            // After you've passed the barrier, update your local flag to make sure that
            // the next time through, you know what to wait on.
            if (gid != first_wg_in_row && lid == 0UL)
                rowBlocks[gid] ^= (1UL << WGBITS);

            // All but the final workgroup in a long-row collaboration have the same start_row
            // and stop_row. They only run for one iteration.
            // Load in a bunch of partial results into your register space, rather than LDS (no contention)
            // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
            const unsigned int col = vecStart + lid;
            if (row == stop_row) // inner thread, we can hardcode/unroll this loop
            {
                // Don't put BLOCK_MULTIPLIER*BLOCKSIZE as the stop point, because
                // some GPU compilers will *aggressively* unroll this loop.
                // That increases register pressure and reduces occupancy.
                for (int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
                {
                    temp_sum = two_fma<T> (alpha*vals[col + j], vec[cols[col + j]], temp_sum, sumk_e);
#if 2*WG_SIZE <= BLOCK_MULTIPLIER*BLOCKSIZE
                    // If you can, unroll this loop once. It somewhat helps performance.
                    j += WG_SIZE;
                    temp_sum = two_fma<T> (alpha*vals[col + j], vec[cols[col + j]], temp_sum, sumk_e);
#endif
                }
            }
            else
            {
                for(int j = 0; j < (int)(vecEnd - col); j += WG_SIZE)
                    temp_sum = two_fma<T> (alpha*vals[col + j], vec[cols[col + j]], temp_sum, sumk_e);
            }

            temp_sum = two_sum<T> (temp_sum, sumk_e, new_error);
            partialSums[lid] = temp_sum;

            // Reduce partial sums
            for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
            {
                tidx.barrier.wait();
                temp_sum = sum2_reduce<T> (temp_sum, new_error, partialSums, lid, lid, i, WG_SIZE, tidx);
            }

            if (lid == 0UL)
            {
                atomic_two_sum_float(&out[row], temp_sum, &new_error);

#ifdef EXTENDED_PRECISION
                unsigned int error_loc = grdExt[0]/tidx.tile_dim[0] + first_wg_in_row + 1;
                // The last half of the rowBlocks buffer is used to hold errors.
                atomic_add_float(&(rowBlocks[error_loc]), new_error);
                // Coordinate across all of the workgroups in this coop in order to have
                // the last workgroup fix up the error values.
                // If this workgroup's row is different than the next workgroup's row
                // then this is the last workgroup -- it's this workgroup's job to add
                // the error values into the final sum.
                if (row != stop_row)
                {
                    // Go forward once your ID is the same as the low order bits of the
                    // coop's first workgroup. That value will be used to store the number
                    // of threads that have completed so far. Once all the previous threads
                    // are done, it's time to send out the errors!
                    while ((hcsparse_atomic_max(&rowBlocks[first_wg_in_row], 0UL) & ((1UL << WGBITS) - 1)) != wg);

                    new_error = (T)(rowBlocks[error_loc]);

                    // Don't need to work atomically here, because this is the only workgroup
                    // left working on this row.
                    out[row] += new_error;
                    rowBlocks[error_loc] = 0UL;

                    // Reset the rowBlocks low order bits for next time.
                    rowBlocks[first_wg_in_row] = rowBlocks[gid] - wg;
                }
                else
                {
                    // Otherwise, increment the low order bits of the first thread in this
                    // coop. We're using this to tell how many workgroups in a coop are done.
                    // Do this with an atomic, since other threads may be doing this too.
                    hcsparse_atomic_inc(&rowBlocks[first_wg_in_row]);
                }
#endif
            }
        }
    }).wait();
}

template <typename T>
hcsparseStatus
csrmv_adaptive( const hcsparseScalar* pAlpha,
                const hcsparseCsrMatrix* pCsrMatx,
                const hcdenseVector* pX,
                const hcsparseScalar* pBeta,
                hcdenseVector* pY,
                hcsparseControl *control )
{
    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of WG_SIZE. Don't know if that
    // have an impact on performance
    // Setting global work size to half the row block size because we are only
    // using half the row blocks buffer for actual work.
    // The other half is used for the extended precision reduction.
    uint global_work_size = ( (pCsrMatx->rowBlockSize/2) - 1 ) * WG_SIZE;

    if( global_work_size < WG_SIZE)
    {
        global_work_size = WG_SIZE;
    }

    T *avCsrMatx_values = static_cast<T*>(pCsrMatx->values);
    int *avColIndices = static_cast<int*>(pCsrMatx->colIndices);
    int *avRowOffsets = static_cast<int*>(pCsrMatx->rowOffsets);
    T *avX_values = static_cast<T*>(pX->values);
    T *avY_values = static_cast<T*>(pY->values);
    unsigned long *avRowBlocks = static_cast<unsigned long*>(pCsrMatx->rowBlocks);
    T *avAlpha = static_cast<T*>(pAlpha->value);
    T *avBeta = static_cast<T*>(pBeta->value);

    csrmv_adaptive_kernel<T> (avCsrMatx_values, avColIndices,
                              avRowOffsets, avX_values,
                              avY_values, avRowBlocks,
                              avAlpha , avBeta, global_work_size, control);

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv (const hcsparseScalar *pAlpha,
       const hcsparseCsrMatrix *pCsrMatx,
       const hcdenseVector *pX,
       const hcsparseScalar *pBeta,
       hcdenseVector *pY,
       hcsparseControl *control)
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

template <typename T>
hcsparseStatus
csrmv ( hcsparseControl *control,
        int m, int n, int nnz, const T *alpha,
        const T *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const T *x, const T *beta,
        T *y)
{
    uint nnz_per_row = nnz/m; //average nnz per row

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
    uint predicted = SUBWAVE_SIZE * m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of WG_SIZE. Don't know if that
    // have an impact on performance
    uint global_work_size =
            WG_SIZE * ((predicted + WG_SIZE - 1 ) / WG_SIZE);

    if( global_work_size < WG_SIZE)
    {
        global_work_size = WG_SIZE;
    }

    csrmv_vector_kernel<T> (m, alpha, 0,
                            csrRowPtrA, csrColIndA, csrValA,
                            x, 0, beta, 0,
                            y, 0, global_work_size, control);

    return hcsparseSuccess;

}
