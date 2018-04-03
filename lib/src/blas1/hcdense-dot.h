#include "hcsparse.h"
#include "reduce-operators.h"
#define BLOCK_SIZE 256

template <typename T>
void inner_product (const long size,
                    T *pR,
                    const long pROffset,
                    T *pX,
                    const long pXOffset,
                    T *pY,
                    const long pYOffset,
                    T *partial,
                    const int REDUCE_BLOCKS_NUMBER,
                    hcsparseControl* control)
{
    hc::extent<1> grdExt(REDUCE_BLOCKS_NUMBER * BLOCK_SIZE);
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view,
                          t_ext,
                          [=](hc::tiled_index<1> tidx) [[hc]]
    {
        tile_static T buf_tmp[BLOCK_SIZE];

        T sum = 0;

	    int eidx = tidx.global[0];
        while(eidx < size)
        {
            sum += pX[pXOffset + eidx] * pY[pYOffset + eidx];
            eidx += REDUCE_BLOCKS_NUMBER * BLOCK_SIZE;
        }
        buf_tmp[tidx.local[0]] = sum;
        tidx.barrier.wait();

        // Seqential part
        if (tidx.local[0] == 0)
        {
            sum = 0;
            for (uint i = 0; i < BLOCK_SIZE; i++)
            {
                sum += buf_tmp[i];
            }
            partial[tidx.tile[0]] = sum;
        }
    }).wait();

    hc::extent<1> grdExt1(1);
    hc::tiled_extent<1> t_ext1 = grdExt1.tile(1);
    hc::parallel_for_each(control->accl_view,
                          t_ext1,
                          [=](hc::tiled_index<1> tidx) [[hc]]
    {
        T sum = 0;
        for (uint i = 0; i < REDUCE_BLOCKS_NUMBER; i++)
        {
            sum += partial[i];
        }
        pR[pROffset] = sum;
    }).wait();
}

template <typename T>
inline
hcsparseStatus dot(hcsparseScalar* pR,
                   const hcdenseVector* pX,
                   const hcdenseVector* pY,
                   hcsparseControl* control)
{
    int size = pX->num_values;
    int REDUCE_BLOCKS_NUMBER = size/BLOCK_SIZE + 1;

    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *partial = (T*) am_alloc(sizeof(T) * REDUCE_BLOCKS_NUMBER, acc, 0);

    T *avR = static_cast<T*>(pR->value);
    T *avX = static_cast<T*>(pX->values);
    T *avY = static_cast<T*>(pY->values);

    inner_product<T>(size,
                     avR,
                     pR->offValue,
                     avX,
                     pX->offValues,
                     avY,
                     pY->offValues,
                     partial, REDUCE_BLOCKS_NUMBER, control);

    control->accl_view.wait();

    am_free(partial);

    return hcsparseSuccess;
}

