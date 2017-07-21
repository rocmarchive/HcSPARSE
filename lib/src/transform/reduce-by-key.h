#include "hcsparse.h"

#define BLOCL_SIZE 256

template <typename T>
hcsparseStatus
reduce_by_key (int size,
               T *keys_output,
               T *values_output,
               const T *keys_input,
               const T *values_input,
               hcsparseControl* control)
{
    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *offsetArray = (T*) am_alloc(size * sizeof(T), acc, 0);
    T *offsetValArray = (T*) am_alloc(size * sizeof(T), acc, 0);

    int numWrkGrp = (size - 1)/BLOCK_SIZE + 1;

    hc::extent<1> grdExt_numElm(BLOCK_SIZE * numWrkGrp);
    hc::tiled_extent<1> t_ext_numElm = grdExt_numElm.tile(BLOCK_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext_numElm, [=] (hc::tiled_index<1> &tidx) [[hc]]
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

    T *keySumArray = (T*) am_alloc(numWrkGrp * sizeof(T), acc, 0);
    T *preSumArray = (T*) am_alloc(numWrkGrp * sizeof(T), acc, 0);
    T *postSumArray = (T*) am_alloc(numWrkGrp * sizeof(T), acc, 0);

    hc::parallel_for_each(control->accl_view, t_ext_numElm, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static T ldsKeys[BLOCK_SIZE];
        tile_static T ldsVals[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim[0];
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

    hc::extent<1> grdExt_blk(BLOCK_SIZE);
    hc::tiled_extent<1> t_ext_blk = grdExt_blk.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext_blk, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static T ldsVals[BLOCK_SIZE];
        tile_static T ldsKeys[BLOCK_SIZE];
        size_t gloId = tidx.global[0];
        size_t locId = tidx.local[0];
        size_t wgSize = tidx.tile_dim[0];
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

    hc::parallel_for_each(control->accl_view, t_ext_numElm, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        size_t gloId = tidx.global[0];
        size_t groId = tidx.tile[0];
        //  Abort threads that are passed the end of the input vector
        if( gloId >= size) return;
        // accumulate prefix
        T key1 = 0;
        if (groId > 0)
          key1 = keySumArray[ groId-1 ];
        T key2 = offsetArray[ gloId ];
        T key3 = -1;
        if(gloId < size -1 )
            key3 =  offsetArray[ gloId + 1];

        if (groId > 0 && key1 == key2 && key2 != key3 )
        {
            T scanResult = offsetValArray[ gloId ];
            T postBlockSum = postSumArray[ groId-1 ];
            T newResult = scanResult + postBlockSum;
            offsetValArray[ gloId ] = newResult;
        }

    });

    hc::parallel_for_each(control->accl_view, t_ext_numElm, [=] (hc::tiled_index<1> &tidx) [[hc]]
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

    control->accl_view.wait();
    am_free(offsetArray);
    am_free(offsetValArray);
    am_free(keySumArray);
    am_free(preSumArray);
    am_free(postSumArray);

    return hcsparseSuccess;
}

