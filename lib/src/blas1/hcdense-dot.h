#include "hcsparse.h"
#include "reduce-operators.h"
#define BLOCK_SIZE 256

template <typename T>
void inner_product (const long size,
                    hc::array_view<T> &pR,
                    const long pROffset,
                    hc::array_view<T> &pX,
                    const long pXOffset,
                    hc::array_view<T> &pY,
                    const long pYOffset,
                    hc::array_view<T> &partial,
                    const int REDUCE_BLOCKS_NUMBER,
                    const hcsparseControl* control)
{
    hc::extent<1> grdExt(REDUCE_BLOCKS_NUMBER * BLOCK_SIZE);
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
    {
        tile_static T buf_tmp[BLOCK_SIZE];
        int idx = tidx.global[0];
        int block_idx = idx / BLOCK_SIZE;
        int thread_in_block_idx = idx % BLOCK_SIZE;
        int eidx = idx;
        T sum = 0;

        while(eidx < size)
        {
            sum = pX[pXOffset + eidx] * pY[pYOffset + eidx];
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

    hc::extent<1> grdExt1(1);
    hc::tiled_extent<1> t_ext1 = grdExt1.tile(1);
    hc::parallel_for_each(control->accl_view, t_ext1, [=] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
    {
        T sum = 0;
        for (uint i = 0; i < REDUCE_BLOCKS_NUMBER; i++)
        {
            sum += partial[i];
        }
        pR[pROffset] = sum;
    });
}

template <typename T>
hcsparseStatus
dot (hcsparseScalar* pR,
     const hcdenseVector* pX,
     const hcdenseVector* pY,
     const hcsparseControl* control)
{
    int size = pX->num_values;
    int REDUCE_BLOCKS_NUMBER = size/BLOCK_SIZE + 1;

    T *partial = (T*) calloc(REDUCE_BLOCKS_NUMBER, sizeof(T));

    hc::array_view<T> *avR = static_cast<hc::array_view<T>*>(pR->value);
    hc::array_view<T> *avX = static_cast<hc::array_view<T>*>(pX->values);
    hc::array_view<T> *avY = static_cast<hc::array_view<T>*>(pY->values);
    hc::array_view<T> avPartial(REDUCE_BLOCKS_NUMBER, partial);

    inner_product<T> (size, *avR, pR->offValue, *avX, pX->offValues, *avY, pY->offValues, avPartial, REDUCE_BLOCKS_NUMBER, control);

    return hcsparseSuccess;
}

