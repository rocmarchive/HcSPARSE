#include "../hcsparse.h"
#include "reduce-operators.h"
#define BLOCK_SIZE 256

template <typename T, ReduceOperator G_OP>
void global_reduce (const long size,
                    Concurrency::array_view<T> &pR,
                    const long pROffset,
                    Concurrency::array_view<T> &pX,
                    const long pXOffset,
                    Concurrency::array_view<T> &partial,
                    const int REDUCE_BLOCKS_NUMBER,
                    const hcsparseControl* control)
{

    Concurrency::extent<1> grdExt(REDUCE_BLOCKS_NUMBER * BLOCK_SIZE);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T buf_tmp[BLOCK_SIZE];
        int idx = tidx.global[0];
        int block_idx = idx / BLOCK_SIZE;
        int thread_in_block_idx = idx % BLOCK_SIZE;
        int eidx = idx;
        T sum = 0;

        while(eidx < size)
        {
            sum = reduceOperation<T, G_OP>(sum, pX[pXOffset + eidx]);
            eidx += REDUCE_BLOCKS_NUMBER * BLOCK_SIZE;
        }
        buf_tmp[thread_in_block_idx] = sum;
        tidx.barrier.wait();

        // Seqential part
        if (tidx.local[0] == 0)
        {
            sum = 0;
            for (uint i = 0; i < BLOCK_SIZE; i++)
            {
                sum += buf_tmp[i];
            }
            partial[block_idx] = sum;
        }
    });

    Concurrency::extent<1> grdExt1(1);
    Concurrency::tiled_extent<1> t_ext1(grdExt1);
    Concurrency::parallel_for_each(control->accl_view, t_ext1, [=] (Concurrency::tiled_index<1> tidx) restrict(amp)
    {
        T sum = 0;
        for (uint i = 0; i < REDUCE_BLOCKS_NUMBER; i++)
        {
            sum += partial[i];
        }
        pR[pROffset] = sum;
    });
}

template<typename T, ReduceOperator G_OP>
hcsparseStatus
reduce(hcsparseScalar* pR,
       const hcdenseVector* pX,
       const hcsparseControl* control)
{
    int size = pX->num_values;
    int REDUCE_BLOCKS_NUMBER = size/BLOCK_SIZE + 1;

    T *partial = (T*) calloc(REDUCE_BLOCKS_NUMBER, sizeof(T));

    Concurrency::array_view<T> *avR = static_cast<Concurrency::array_view<T>*>(pR->value);
    Concurrency::array_view<T> *avX = static_cast<Concurrency::array_view<T>*>(pX->values);
    Concurrency::array_view<T> avPartial(REDUCE_BLOCKS_NUMBER, partial);

    global_reduce<T, G_OP> (size, *avR, pR->offValue, *avX, pX->offValues, avPartial, REDUCE_BLOCKS_NUMBER, control);

    return hcsparseSuccess;
}

