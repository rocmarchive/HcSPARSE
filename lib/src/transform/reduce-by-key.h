#include "hcsparse.h"

#define BLOCL_SIZE 256

template <typename T>
hcsparseStatus
reduce_by_key( int size,
               Concurrency::array_view<T> &keys_output, 
               Concurrency::array_view<T> &values_output,
               const Concurrency::array_view<T> &keys_input, 
               const Concurrency::array_view<T> &values_input,
               const hcsparseControl* control)
{
    //this vector stores the places where input index is changing;
    T* offsetArray_buff = (T*) calloc (size, sizeof(T));
    T* offsetValArray_buff = (T*) calloc (size, sizeof(T));

    Concurrency::array_view<T> offsetArray(size, offsetArray_buff);
    Concurrency::array_view<T> offsetValArray(size, offsetValArray_buff);

    int numWrkGrp = (size - 1)/BLOCK_SIZE + 1;

    Concurrency::extent<1> grdExt_numElm(BLOCK_SIZE * numWrkGrp);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext_numElm(grdExt_numElm);

    Concurrency::parallel_for_each(control->accl_view, t_ext_numElm, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        size_t gloId = tidx.global[0];
        if (gloId >= size) return;
        T key, prev_key;
        if(gloId > 0)
        {
            key = keys_input[ gloId ];
            prev_key = keys_input[ gloId - 1];
            if(key == prev_key)
                offsetArray[ gloId ] = 0;
            else
                offsetArray[ gloId ] = 1;
        }
        else
        {
             offsetArray[ gloId ] = 0;
        }
    });

    inclusive_scan<T, EW_PLUS>(size, offsetArray, offsetArray, control);

    T* keySumArray_buff = (T*) calloc (numWrkGrp, sizeof(T));
    T* preSumArray_buff = (T*) calloc (numWrkGrp, sizeof(T));
    T* postSumArray_buff = (T*) calloc (numWrkGrp, sizeof(T));

    Concurrency::array_view<T> keySumArray(numWrkGrp, keySumArray_buff);
    Concurrency::array_view<T> preSumArray(numWrkGrp, preSumArray_buff);
    Concurrency::array_view<T> postSumArray(numWrkGrp, postSumArray_buff);

    Concurrency::parallel_for_each(control->accl_view, t_ext_numElm, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T ldsKeys[BLOCK_SIZE];
        tile_static T ldsVals[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim0;
        T key;
        T val = 0;
        if(gloId < size)
        {
            key = offsetArray[ gloId ];
            val = values_input[ gloId ];
            ldsKeys[ locId ] = key;
            ldsVals[ locId ] = val;
        }
        else
        {
            ldsKeys[ locId ] = offsetArray[size-1];
            ldsVals[ locId ] = 0;
        }
        // Computes a scan within a workgroup
        // updates vals in lds but not keys
        T sum = val;
        for( size_t offset = 1; offset < wgSize; offset *= 2 )
        {
            tidx.barrier.wait();
            T key2 = ldsKeys[locId - offset];
            if (locId >= offset && key == key2)
            {
                T y = ldsVals[ locId - offset ];
                sum = sum + y;
            }
            tidx.barrier.wait();
            ldsVals[ locId ] = sum;
        }
        tidx.barrier.wait();
        //  Abort threads that are passed the end of the input vector
        if (gloId >= size) return;
        // Each work item writes out its calculated scan result, relative to the beginning
        // of each work group
        T key2 = -1;
        if (gloId < size -1 )
            key2 = offsetArray[gloId + 1];
        if(key != key2)
           offsetValArray[ gloId ] = sum;
        if (locId == 0)
        {
            keySumArray[ groId ] = ldsKeys[ wgSize-1 ];
            preSumArray[ groId ] = ldsVals[ wgSize-1 ];
        }
    });

    int workPerThread = (numWrkGrp - 1) / BLOCK_SIZE + 1;

    Concurrency::extent<1> grdExt_blk(BLOCK_SIZE);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext_blk(grdExt_blk);

    Concurrency::parallel_for_each(control->accl_view, t_ext_blk, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        tile_static T ldsVals[BLOCK_SIZE];
        tile_static T ldsKeys[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim0;
        uint mapId  = gloId * workPerThread;
        // do offset of zero manually
        int offset;
        T key = 0;
        T workSum = 0;
        if (mapId < numWrkGrp)
        {
            T prevKey;
            // accumulate zeroth value manually
            offset = 0;
            key = keySumArray[ mapId+offset ];
            workSum = preSumArray[ mapId+offset ];
            postSumArray[ mapId+offset ] = workSum;
            //  Serial accumulation
            for( offset = offset+1; offset < workPerThread; offset += 1 )
            {
                prevKey = key;
                key = keySumArray[ mapId+offset ];
                if (mapId+offset < numWrkGrp)
                {
                    T y = preSumArray[ mapId+offset ];
                    if (key == prevKey)
                    {
                        workSum = workSum + y;
                    }
                    else
                    {
                        workSum = y;
                    }
                    postSumArray[ mapId+offset ] = workSum;
                }
            }
        }
        tidx.barrier.wait();
        T scanSum = workSum;
        offset = 1;
        // load LDS with register sums
        ldsVals[ locId ] = workSum;
        ldsKeys[ locId ] = key;
        // scan in lds
        for( offset = offset*1; offset < wgSize; offset *= 2 )
        {
            tidx.barrier.wait();
            if (mapId < numWrkGrp)
            {
                if (locId >= offset  )
                {
                    T y    = ldsVals[ locId - offset ];
                    T key1 = ldsKeys[ locId ];
                    T key2 = ldsKeys[ locId-offset ];
                    if ( key1 == key2 )
                    {
                        scanSum = scanSum + y;
                    }
                    else
                        scanSum = ldsVals[ locId ];
                 }
            }
            tidx.barrier.wait();
            ldsVals[ locId ] = scanSum;
        } // for offset
        tidx.barrier.wait();
        // write final scan from pre-scan and lds scan
        for( offset = 0; offset < workPerThread; offset += 1 )
        {
            tidx.barrier.wait();
            if (mapId < numWrkGrp && locId > 0)
            {
                T y    = postSumArray[ mapId+offset ];
                T key1 = keySumArray[ mapId+offset ]; // change me
                T key2 = ldsKeys[ locId-1 ];
                if ( key1 == key2 )
                {
                    T y2 = ldsVals[locId-1];
                    y = y + y2;
                }
                postSumArray[ mapId+offset ] = y;
            } 
        } 
    }); 

    Concurrency::parallel_for_each(control->accl_view, t_ext_numElm, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        //  Abort threads that are passed the end of the input vector
        if( gloId >= size) return;
        // accumulate prefix
        T key1 = keySumArray[ groId-1 ];
        T key2 = offsetArray[ gloId ];
        T key3 = -1;
        if(gloId < size -1 )
            key3 =  offsetArray[ gloId + 1];
        if (groId > 0 && key1 == key2 && key2 != key3)
        {
            T scanResult = offsetValArray[ gloId ];
            T postBlockSum = postSumArray[ groId-1 ];
            T newResult = scanResult + postBlockSum;
            offsetValArray[ gloId ] = newResult;
        }
    });

    Concurrency::parallel_for_each(control->accl_view, t_ext_numElm, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        size_t gloId = tidx.global[0];
        //  Abort threads that are passed the end of the input vector
        if( gloId >= size) return;
        int numSections = offsetArray[size-1] + 1;
        if(gloId < (size-1) && offsetArray[ gloId ] != offsetArray[ gloId +1])
        {
            keys_output[ offsetArray [ gloId ]] = keys_input[ gloId];
            values_output[ offsetArray [ gloId ]] = offsetValArray [ gloId];
        }
        if( gloId == (size-1) )
        {
            keys_output[ numSections - 1 ] = keys_input[ gloId ]; //Copying the last key directly. Works either ways
            values_output[ numSections - 1 ] = offsetValArray [ gloId ];
            offsetArray [ gloId ] = numSections;
        }
    });

    return hcsparseSuccess;

}
