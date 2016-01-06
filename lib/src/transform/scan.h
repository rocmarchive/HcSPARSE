#include "hcsparse.h"

#define BLOCK_SIZE 256

template <typename T, ElementWiseOperator OP>
hcsparseStatus
scan(int size,
     Concurrency::array_view<T> &output, 
     const Concurrency::array_view<T> &input,
     const hcsparseControl* control, 
     bool exclusive)
{
    T* preSumArray_buff = (T*) calloc (size, sizeof(T));
    T* preSumArray1_buff = (T*) calloc (size, sizeof(T));
    T* postSumArray_buff = (T*) calloc (size, sizeof(T));

    Concurrency::array_view<T> preSumArray(size, preSumArray_buff);
    Concurrency::array_view<T> preSumArray1(size, preSumArray1_buff);
    Concurrency::array_view<T> postSumArray(size, postSumArray_buff);

    T identity = 0;

    int numWrkGrp = (size - 1)/BLOCK_SIZE + 1;

    //scan in blocks

    Concurrency::extent<1> grdExt_numElm(BLOCK_SIZE * numWrkGrp);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext_numElm(grdExt_numElm);

    Concurrency::parallel_for_each(control->accl_view, t_ext_numElm, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T lds[BLOCK_SIZE*2];
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim0;
        wgSize *=2;
        size_t offset = 1;
        // load input into shared memory
        if(groId * wgSize + locId < size)
            lds[locId] = input[groId * wgSize + locId];
        else
            lds[locId] = 0;//input_ptr[vecSize - 1];
        if(groId * wgSize + locId + (wgSize / 2) < size)
            lds[locId + (wgSize / 2)] = input[groId * wgSize + locId + (wgSize / 2)];
        else
            lds[locId + (wgSize / 2)] = 0;
        // Exclusive case
        if(exclusive && gloId == 0)
        {
            T start_val = input[0];
            lds[locId] = operation<T, OP>(identity, start_val);
        }
        for (size_t start = wgSize>>1; start > 0; start >>= 1)
        {
            tidx.barrier.wait();
            if (locId < start)
            {
                size_t temp1 = offset*(2*locId+1)-1;
                size_t temp2 = offset*(2*locId+2)-1;
                T y = lds[temp2];
                T y1 =lds[temp1];
                lds[temp2] = operation<T, OP>(y, y1);
            }
            offset *= 2;
        }
        tidx.barrier.wait();
        if (locId == 0)
        {
            preSumArray[ groId ] = lds[wgSize -1];
            preSumArray1[ groId ] = lds[wgSize/2 -1];
        }
    });

    T workPerThread = size / BLOCK_SIZE;

    Concurrency::extent<1> grdExt_block(BLOCK_SIZE);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext_block(grdExt_block);

    Concurrency::parallel_for_each(control->accl_view, t_ext_block, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T lds[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim0;
        uint mapId  = gloId * workPerThread;
        // do offset of zero manually
        uint offset;
        T workSum = 0;
         if (mapId < size)
        {
            // accumulate zeroth value manually
            offset = 0;
            workSum = preSumArray[mapId+offset];
            //  Serial accumulation
            for ( offset = offset + 1; offset < workPerThread; offset += 1 )
            {
                if (mapId + offset < size)
                {
                    T y = preSumArray[mapId+offset];
                    workSum = operation<T, OP>(workSum,y);
                }
            }
        }
        tidx.barrier.wait();
        T scanSum = workSum;
        lds[ locId ] = workSum;
        offset = 1;
        // scan in lds
        for ( offset = offset*1; offset < wgSize; offset *= 2 )
        {
            tidx.barrier.wait();
            if (mapId < size)
            {
                if (locId >= offset)
                {
                    T y = lds[ locId - offset ];
                    scanSum = operation<T, OP>(scanSum, y);
                }
            }
            tidx.barrier.wait();
            lds[ locId ] = scanSum;
        } // for offset
        tidx.barrier.wait();
        // write final scan from pre-scan and lds scan
        workSum = preSumArray[mapId];
        if (locId > 0)
        {
            T y = lds[locId-1];
            workSum = operation<T, OP>(workSum, y);
            postSumArray[ mapId] = workSum;
        }
        else
        {
            postSumArray[ mapId] = workSum;
        }
        for ( offset = 1; offset < workPerThread; offset += 1 )
        {
            tidx.barrier.wait();
            if (mapId < size && locId > 0)
            {
                T y  = preSumArray[ mapId + offset ] ;
                T y1 = operation<T, OP>(y, workSum);
                postSumArray[ mapId + offset ] = y1;
                workSum = y1;
            } // thread in bounds
            else
            {
                T y  = preSumArray[ mapId + offset ] ;
                postSumArray[ mapId + offset ] = operation<T, OP>(y, workSum);
                workSum = postSumArray[ mapId + offset ];
            }
        } // for
    });

    Concurrency::parallel_for_each(control->accl_view, t_ext_block, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T lds[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim0;
        // if exclusive, load gloId=0 w/ identity, and all others shifted-1
        T val;
        if (gloId < size)
        {
            if (exclusive)
            {
                if (gloId > 0)
                {
                    val = input[gloId-1];
                    lds[ locId ] = val;
                }
                else
                {
                    val = identity;
                    lds[ locId ] = val;
                }
            }
            else
            {
                val = input[gloId];
                lds[ locId ] = val;
            }
        }
        T scanResult = lds[locId];
        T postBlockSum, newResult;
        T y, y1, sum;
        if (locId == 0 && gloId < size)
        {
            if (groId > 0)
            {
                if (groId % 2 == 0)
                    postBlockSum = postSumArray[ groId/2 -1 ];
                else if (groId == 1)
                    postBlockSum = preSumArray[0];
                else
                {
                    y = postSumArray[ groId/2 -1 ];
                    y1 = preSumArray[groId/2];
                    postBlockSum = operation<T, OP>(y, y1);
                }
                if (exclusive)
                    newResult = postBlockSum;
                else
                    newResult = operation<T, OP>(scanResult, postBlockSum);
            }
            else
            {
                newResult = scanResult;
            }
            lds[ locId ] = newResult;
        }
        //  Computes a scan within a workgroup
        sum = lds[ locId ];
        for ( size_t offset = 1; offset < wgSize; offset *= 2 )
        {
            tidx.barrier.wait();
            if (locId >= offset)
            {
                T y = lds[ locId - offset ];
                sum = operation<T, OP>(sum, y);
            }
            tidx.barrier.wait();
            lds[ locId ] = sum;
        }
        tidx.barrier.wait();
        //  Abort threads that are passed the end of the input vector
        if (gloId >= size) return;
        output[ gloId ] = sum;
    });

    return hcsparseSuccess;
}

template <typename T, ElementWiseOperator OP>
hcsparseStatus
exclusive_scan( int size, 
                Concurrency::array_view<T> &output,
                const Concurrency::array_view<T> &input,
                const hcsparseControl* control)
{
   return scan<T, OP>(size, output, input, control, true);
}

template <typename T, ElementWiseOperator OP>
hcsparseStatus
inclusive_scan( int size,
                Concurrency::array_view<T> &output,
                const Concurrency::array_view<T> &input,
                const hcsparseControl* control)
{
  return scan<T, OP>(size, output, input, control, false);
}


