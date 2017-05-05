#include "hcsparse.h"
#include "reduce-operators.h"
#define BLOCK_SIZE 256

template <typename T, ReduceOperator G_OP, ReduceOperator F_OP = RO_DUMMY>
void reduce_row_column (T *pX,
                        T *partial,
                        const int m,
                        const int n,
                        hcsparseControl* control)
{

    hc::extent<1> grdExt( m * BLOCK_SIZE);
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) [[hc]]
    {
        tile_static T buf_tmp[BLOCK_SIZE];
        uint idx = tidx.global[0];
        uint lidx = tidx.local[0];
        uint eidx = lidx;
        T sum = 0;

        while (eidx < n) {
          sum += pX[tidx.tile[0] * n + eidx];
          eidx += BLOCK_SIZE;
        }
        buf_tmp[lidx] = sum;
        tidx.barrier.wait();

        if (lidx == 0) {
          sum = 0;
          for (uint i = 0; i < n; i++) { 
            sum += buf_tmp[i]; 
          }
          partial[tidx.tile[0]] = sum;
        }
    });
}

template <typename T, ReduceOperator G_OP, ReduceOperator F_OP>
void global_reduce (const long size,
                    T *pR,
                    const long pROffset,
                    T *pX,
                    const long pXOffset,
                    T *partial,
                    const int REDUCE_BLOCKS_NUMBER,
                    hcsparseControl* control)
{

    hc::extent<1> grdExt(REDUCE_BLOCKS_NUMBER * BLOCK_SIZE);
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) [[hc]]
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

    hc::extent<1> grdExt1(1);
    hc::tiled_extent<1> t_ext1 = grdExt1.tile(1);
    hc::parallel_for_each(control->accl_view, t_ext1, [=] (hc::tiled_index<1>& tidx) [[hc]]
    {
        T sum = 0;
        for (uint i = 0; i < REDUCE_BLOCKS_NUMBER; i++)
        {
            sum += partial[i];
        }
        pR[pROffset] = reduceOperation<T, F_OP>(sum);
    });
}

template <typename T, ReduceOperator G_OP, ReduceOperator F_OP = RO_DUMMY>
hcsparseStatus
reduce (hcsparseScalar* pR,
        const hcdenseVector* pX,
        hcsparseControl* control)
{
    int size = pX->num_values;
    int REDUCE_BLOCKS_NUMBER = size/BLOCK_SIZE + 1;

    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *partial = (T*) am_alloc(sizeof(T) * REDUCE_BLOCKS_NUMBER, acc, 0);

    T *avR = static_cast<T*>(pR->value);
    T *avX = static_cast<T*>(pX->values);

    global_reduce<T, G_OP, F_OP> (size, avR, pR->offValue, avX, pX->offValues, partial, REDUCE_BLOCKS_NUMBER, control);

    return hcsparseSuccess;
}

