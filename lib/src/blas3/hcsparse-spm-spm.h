#include "hcsparse.h"

#define GROUPSIZE_256 256
#define TUPLE_QUEUE 6
#define NUM_SEGMENTS 128
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int 
#define MERGEPATH_LOCAL     0
#define MERGEPATH_LOCAL_L2  1
#define MERGEPATH_GLOBAL    2
#define MERGELIST_INITSIZE 256
#define BHSPARSE_SUCCESS 0
 
int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m);

template <typename T>
inline
void siftDown(int *s_key,
              T   *s_val,
              const int start,
              const int stop,
              const int local_id,
              const int local_size) restrict (amp) 
{
    int root = start;
    int child, swap;
    int temp_swap_key;
    T temp_swap_val;
    while (root * 2 + 1 <= stop)
    {
        child = root * 2 + 1;
        swap = root;
        if (s_key[swap * local_size + local_id] < s_key[child * local_size + local_id])
            swap = child;
        if (child + 1 <= stop && s_key[swap * local_size + local_id] < s_key[(child + 1) * local_size + local_id])
            swap = child + 1;
        if (swap != root)
        {
            const int index1 = root * local_size + local_id;
            const int index2 = swap * local_size + local_id;
            //swap root and swap
            temp_swap_key = s_key[index1];
            s_key[index1] = s_key[index2];
            s_key[index2] = temp_swap_key;
            temp_swap_val = s_val[index1];
            s_val[index1] = s_val[index2];
            s_val[index2] = temp_swap_val;
            root = swap;
        }
        else
            return;
    }
}

template <typename T>
inline
int heapsort(int *s_key,
             T   *s_val,
             const int segment_size,
             const int local_id,
             const int local_size) restrict (amp)
{
    // heapsort - heapify max-heap
    int start = (segment_size - 1) / 2;
    int stop  = segment_size - 1;
    int index1, index2;
    while (start >= 0)
    {
        siftDown(s_key, s_val, start, stop, local_id, local_size);
        start--;
    }
    // inject root element to the end
    int temp_swap_key;
    T temp_swap_val;
    index1 = stop * local_size + local_id;
    temp_swap_key = s_key[local_id];
    s_key[local_id] = s_key[index1];
    s_key[index1] = temp_swap_key;
    temp_swap_val = s_val[local_id];
    s_val[local_id] = s_val[index1];
    s_val[index1] = temp_swap_val;
    stop--;
    siftDown(s_key, s_val, 0, stop, local_id, local_size);
    // this start is compressed list's start
    start = segment_size - 1;
    // heapsort - remove-max and compress
    while (stop >= 0)
    {
        index2 = stop * local_size + local_id;
        if (s_key[local_id] == s_key[start * local_size + local_id])
        {
            s_val[start * local_size + local_id] += s_val[local_id];
            s_key[local_id] = s_key[index2];
            s_val[local_id] = s_val[index2];
        }
        else
        {
            start--;
            index1 = start * local_size + local_id;
            if (stop == start)
            {
                temp_swap_key = s_key[local_id];
                s_key[local_id] = s_key[index2];
                s_key[index2] = temp_swap_key;
                temp_swap_val = s_val[local_id];
                s_val[local_id] = s_val[index2];
                s_val[index2] = temp_swap_val;
            }
            else
            {
                s_key[index1] = s_key[local_id];
                s_val[index1] = s_val[local_id];
                s_key[local_id] = s_key[index2];
                s_val[local_id] = s_val[index2];
            }
        }
        stop--;
        siftDown(s_key, s_val, 0, stop, local_id, local_size);
    }
    return start;
}

template <typename T>
inline
void coex(int  *keyA,
          T    *valA,
          int  *keyB,
          T    *valB,
          const int  dir)
{
    int t;
    T v;
    if ((*keyA > *keyB) == dir)
    {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
        v = *valA;
        *valA = *valB;
        *valB = v;
    }
}

template <typename T>
inline
void bitonic(int   *s_key,
             T     *s_val,
             int   stage,
             int   passOfStage,
             int   local_id,
             int   local_counter)
{
    int sortIncreasing = 1;
    int pairDistance = 1 << (stage - passOfStage);
    int blockWidth   = 2 * pairDistance;
    int leftId = (local_id % pairDistance) + (local_id / pairDistance) * blockWidth;
    int rightId = leftId + pairDistance;
    int sameDirectionBlockWidth = 1 << stage;
    if((local_id/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;
    int  leftElement  = s_key[leftId];          // index_type
    int  rightElement = s_key[rightId];         // index_type
    T   leftElement_val  = s_val[leftId];      // value_type
    T   rightElement_val = s_val[rightId];     // value_type
    int  greater;         // index_type
    int  lesser;          // index_type
    T   greater_val;     // value_type
    T   lesser_val;      // value_type
    if(leftElement > rightElement)
    {
        greater = leftElement;
        lesser  = rightElement;
        greater_val = leftElement_val;
        lesser_val  = rightElement_val;
    }
    else
    {
        greater = rightElement;
        lesser  = leftElement;
        greater_val = rightElement_val;
        lesser_val  = leftElement_val;
    }
    if(sortIncreasing)
    {
        s_key[leftId]  = lesser;
        s_key[rightId] = greater;
        s_val[leftId]  = lesser_val;
        s_val[rightId] = greater_val;
    }
    else
    {
        s_key[leftId]  = greater;
        s_key[rightId] = lesser;
        s_val[leftId]  = greater_val;
        s_val[rightId] = lesser_val;
    }
}

template <typename T>
inline
void bitonicsort(Concurrency::tiled_index<GROUPSIZE_256> &tidx,
                 int  *s_key,
                 T    *s_val,
                 int  arrayLength) restrict (amp)
{
    int local_id = tidx.local[0];
    int numStages = 0;
    for(int temp = arrayLength; temp > 1; temp >>= 1)
    {
        ++numStages;
    }
    for (int stage = 0; stage < numStages; ++stage)
    {
        for (int passOfStage = 0; passOfStage <= stage; ++passOfStage)
        {
            bitonic<T> (s_key, s_val, stage, passOfStage, local_id, arrayLength);
            tidx.barrier.wait();
        }
    }
}

inline
void scan_512(Concurrency::tiled_index<GROUPSIZE_256> &tidx,
              int *s_scan)
{
    int local_id = tidx.local[0];
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    int temp;
    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    tidx.barrier.wait();
    if (local_id < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 64)  { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[511] += s_scan[255]; s_scan[512] = s_scan[511]; s_scan[511] = 0; temp = s_scan[255]; s_scan[255] = 0; s_scan[511] += temp; }
    if (local_id < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 64) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}

template <typename T>
inline
void compression_scan(Concurrency::tiled_index<GROUPSIZE_256> &tidx,
                      int *s_scan,
                      int *s_key,
                      T   *s_val,
                      const int       local_counter,
                      const int       local_size,
                      const int       local_id,
                      const int       local_id_halfwidth) restrict (amp)
{
    // compression - prefix sum
    bool duplicate = 1;
    bool duplicate_halfwidth = 1;
    // generate bool value in registers
    if (local_id < local_counter && local_id > 0)
        duplicate = (s_key[local_id] != s_key[local_id - 1]);
    if (local_id_halfwidth < local_counter)
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);
    // copy bool values from register to local memory (s_scan)
    s_scan[local_id]                    = duplicate;
    s_scan[local_id_halfwidth]          = duplicate_halfwidth;
    tidx.barrier.wait();
    // in-place exclusive prefix-sum scan on s_scan
//    scan_512(tidx, s_scan);
    tidx.barrier.wait();
    // compute final position and final value in registers
    int   move_pointer;
    int   final_position, final_position_halfwidth;
    int   final_key,      final_key_halfwidth;
    T final_value,    final_value_halfwidth;
    if (local_id < local_counter && duplicate == 1)
    {
        final_position = s_scan[local_id];
        final_key = s_key[local_id];
        final_value = s_val[local_id];
        move_pointer = local_id + 1;
        while (s_scan[move_pointer] == s_scan[move_pointer + 1])
        {
            final_value += s_val[move_pointer];
            move_pointer++;
        }
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        final_position_halfwidth = s_scan[local_id_halfwidth];
        final_key_halfwidth = s_key[local_id_halfwidth];
        final_value_halfwidth = s_val[local_id_halfwidth];
        move_pointer = local_id_halfwidth + 1;
        while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
        {
            final_value_halfwidth += s_val[move_pointer];
            move_pointer++;
        }
    }
    tidx.barrier.wait();
    // write final_positions and final_values to s_key and s_val
    if (local_id < local_counter && duplicate == 1)
    {
        s_key[final_position] = final_key;
        s_val[final_position] = final_value;
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        s_key[final_position_halfwidth] = final_key_halfwidth;
        s_val[final_position_halfwidth] = final_value_halfwidth;
    }
}

template <typename T>
hcsparseStatus compute_nnzCt(int m, 
                             Concurrency::array_view<int> &csrRowPtrA, 
                             Concurrency::array_view<int> &csrColIndA, 
                             Concurrency::array_view<int> &csrRowPtrB, 
                             Concurrency::array_view<int> &csrColIndB, 
                             Concurrency::array_view<int> &csrRowPtrCt, 
                             const hcsparseControl* control)
{  
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    int num_threads = GROUPSIZE_256;
    int num_blocks = Concurrency::fast_math::ceil((double)m / (double)num_threads);

    szLocalWorkSize  = num_threads;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int global_id = tidx.global[0];
        int start, stop, index, strideB, row_size_Ct = 0;
        if (global_id < m)
        {
            start = csrRowPtrA[global_id];
            stop = csrRowPtrA[global_id + 1];
            for (int i = start; i < stop; i++)
            {
                index = csrColIndA[i];
                strideB = csrRowPtrB[index + 1] - csrRowPtrB[index];
                row_size_Ct += strideB;
            }
            csrRowPtrCt[global_id] = row_size_Ct;
        }
        if (global_id == 0)
            csrRowPtrCt[m] = 0;
    });

    return hcsparseSuccess;
 }
 

template <typename T> 
hcsparseStatus compute_nnzC_Ct_0(int num_blocks, int j, 
                                 int counter, int position, 
                                 Concurrency::array_view<int> &queue_one, 
                                 Concurrency::array_view<int> &csrRowPtrC, 
                                 const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;
    
    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int global_id = tidx.global[0];
        if (global_id < counter)
        {
            int row_id = queue_one[TUPLE_QUEUE * (position + global_id)];
            csrRowPtrC[row_id] = 0;
        }
    });

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_1(int num_blocks, int j, 
                                 int counter, int position, 
                                 Concurrency::array_view<int> &queue_one, 
                                 Concurrency::array_view<int> &csrRowPtrA, 
                                 Concurrency::array_view<int> &csrColIndA, 
                                 Concurrency::array_view<T> &csrValA, 
                                 Concurrency::array_view<int> &csrRowPtrB, 
                                 Concurrency::array_view<int> &csrColIndB, 
                                 Concurrency::array_view<T> &csrValB, 
                                 Concurrency::array_view<int> &csrRowPtrC, 
                                 Concurrency::array_view<int> &csrRowPtrCt, 
                                 Concurrency::array_view<int> &csrColIndCt, 
                                 Concurrency::array_view<T> &csrValCt, 
                                 const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int global_id = tidx.global[0];
        if (global_id < counter)
        {
            int row_id = queue_one[TUPLE_QUEUE * (position + global_id)];
            csrRowPtrC[row_id] = 1;
            int base_index = queue_one[TUPLE_QUEUE * (position + global_id) + 1];
            int col_index_A_start = csrRowPtrA[row_id];
            int col_index_A_stop = csrRowPtrA[row_id+1];
            for (int col_index_A = col_index_A_start; col_index_A < col_index_A_stop; col_index_A++)
            {
                int row_id_B = csrColIndA[col_index_A];
                int col_index_B = csrRowPtrB[row_id_B];
                if (col_index_B == csrRowPtrB[row_id_B+1])
                    continue;
                T value_A  = csrValA[col_index_A];
                csrColIndCt[base_index] = csrColIndB[col_index_B];
                csrValCt[base_index] = csrValB[col_index_B] * value_A;
                break;
            }
        }
    });

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_2heap_noncoalesced_local(int num_blocks, int j, 
                                                        int counter, int position, 
                                                        Concurrency::array_view<int> &queue_one, 
                                                        Concurrency::array_view<int> &csrRowPtrA, 
                                                        Concurrency::array_view<int> &csrColIndA, 
                                                        Concurrency::array_view<T> &csrValA, 
                                                        Concurrency::array_view<int> &csrRowPtrB, 
                                                        Concurrency::array_view<int> &csrColIndB, 
                                                        Concurrency::array_view<T> &csrValB, 
                                                        Concurrency::array_view<int> &csrRowPtrC, 
                                                        Concurrency::array_view<int> &csrRowPtrCt, 
                                                        Concurrency::array_view<int> &csrColIndCt, 
                                                        Concurrency::array_view<T> &csrValCt, 
                                                        const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        tile_static int s_key[32*GROUPSIZE_256];
        tile_static T s_val[32*GROUPSIZE_256];
        const int local_id = tidx.local[0];
        const int group_id = tidx.tile[0];
        const int global_id = tidx.global[0];
        const int local_size = tidx.tile_dim0;
        int index = 0;
        if (global_id < counter)
        {
            int i, counter_local = 0;
            int start_col_index_A, stop_col_index_A;
            int rowidB, start_col_index_B, stop_col_index_B;
            T value_A;
            int rowidC = queue_one[TUPLE_QUEUE * (position + global_id)];
            start_col_index_A = csrRowPtrA[rowidC];
            stop_col_index_A  = csrRowPtrA[rowidC + 1];
            // i is both col index of A and row index of B
            for (i = start_col_index_A; i < stop_col_index_A; i++)
            {
                rowidB = csrColIndA[i];
                value_A  = csrValA[i];
                start_col_index_B = csrRowPtrB[rowidB];
                stop_col_index_B  = csrRowPtrB[rowidB + 1];
                for (int j = start_col_index_B; j < stop_col_index_B; j++)
                {
                    index = counter_local * local_size + local_id;
                    s_key[index] = csrColIndB[j];
                    s_val[index] = csrValB[j] * value_A;
                    counter_local++;
                }
            }
            // heapsort in each work-item
            int local_start = heapsort<T> (s_key, s_val, counter_local, local_id, local_size);
            counter_local -= local_start;
            csrRowPtrC[rowidC] = counter_local;
            const int base_index = queue_one[TUPLE_QUEUE * (position + group_id * local_size + local_id) + 1];;
            for (int i = 0; i < counter_local; i++)
            {
                csrColIndCt[base_index + i] = s_key[(local_start+i) * local_size + local_id];
                csrValCt[base_index + i] = s_val[(local_start+i) * local_size + local_id];
            }
        }
    });

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_bitonic_scan(int num_blocks, int j, int position, 
                                            Concurrency::array_view<int> &queue_one, 
                                            Concurrency::array_view<int> &csrRowPtrA, 
                                            Concurrency::array_view<int> &csrColIndA, 
                                            Concurrency::array_view<T> &csrValA, 
                                            Concurrency::array_view<int> &csrRowPtrB, 
                                            Concurrency::array_view<int> &csrColIndB, 
                                            Concurrency::array_view<T> &csrValB, 
                                            Concurrency::array_view<int> &csrRowPtrC, 
                                            Concurrency::array_view<int> &csrRowPtrCt, 
                                            Concurrency::array_view<int> &csrColIndCt, 
                                            Concurrency::array_view<T> &csrValCt, 
                                            int n, 
                                            const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        tile_static int s_key[2*GROUPSIZE_256];
        tile_static T s_val[2*GROUPSIZE_256];
        tile_static int s_scan[2*GROUPSIZE_256 + 1];
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int local_size = tidx.tile_dim0;
        int width = local_size * 2;
        int i, local_counter = 0;
        int strideB, local_offset, global_offset;
        int invalid_width;
        int local_id_halfwidth = local_id + local_size;
        int row_id_B; // index_type
        int row_id;// index_type
        row_id = queue_one[TUPLE_QUEUE * (position + group_id)];
        int start_col_index_A, stop_col_index_A;  // index_type
        int start_col_index_B, stop_col_index_B;  // index_type
        T value_A;                            // value_type
        start_col_index_A = csrRowPtrA[row_id];
        stop_col_index_A  = csrRowPtrA[row_id + 1];
        // i is both col index of A and row index of B
        for (i = start_col_index_A; i < stop_col_index_A; i++)
        {
            row_id_B = csrColIndA[i];
            value_A  = csrValA[i];
            start_col_index_B = csrRowPtrB[row_id_B];
            stop_col_index_B  = csrRowPtrB[row_id_B + 1];
            strideB = stop_col_index_B - start_col_index_B;
            if (local_id < strideB)
            {
                local_offset = local_counter + local_id;
                global_offset = start_col_index_B + local_id;
                s_key[local_offset] = csrColIndB[global_offset];
                s_val[local_offset] = csrValB[global_offset] * value_A;
            }
            if (local_id_halfwidth < strideB)
            {
                local_offset = local_counter + local_id_halfwidth;
                global_offset = start_col_index_B + local_id_halfwidth;
                s_key[local_offset] = csrColIndB[global_offset];
                s_val[local_offset] = csrValB[global_offset] * value_A;
            }
            local_counter += strideB;
        }
        tidx.barrier.wait();
        invalid_width = width - local_counter;
        // to meet 2^N, set the rest elements to n (number of columns of C)
        if (local_id < invalid_width)
            s_key[local_counter + local_id] = n;
        tidx.barrier.wait();
        // bitonic sort
        bitonicsort<T> (tidx, s_key, s_val, width);
        tidx.barrier.wait();
        // compression - scan
        compression_scan(tidx, s_scan, s_key, s_val, local_counter,
                         local_size, local_id, local_id_halfwidth);
        tidx.barrier.wait();
        local_counter = s_scan[width] - invalid_width;
        if (local_id == 0)
            csrRowPtrC[row_id] = local_counter;
        // write compressed lists to global mem
        int row_offset = queue_one[TUPLE_QUEUE * (position + group_id) + 1];
        if (local_id < local_counter)
        {
            global_offset = row_offset + local_id;
            csrColIndCt[global_offset] = s_key[local_id];
            csrValCt[global_offset] = s_val[local_id];
        }
        if (local_id_halfwidth < local_counter)
        {
            global_offset = row_offset + local_id_halfwidth;
            csrColIndCt[global_offset] = s_key[local_id_halfwidth];
            csrValCt[global_offset] = s_val[local_id_halfwidth];
        }
    });
    
    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_mergepath(int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location, 
                                         Concurrency::array_view<int> &queue_one, 
                                         Concurrency::array_view<int> &csrRowPtrA, 
                                         Concurrency::array_view<int> &csrColIndA, 
                                         Concurrency::array_view<T> &csrValA, 
                                         Concurrency::array_view<int> &csrRowPtrB, 
                                         Concurrency::array_view<int> &csrColIndB, 
                                         Concurrency::array_view<T> &csrValB, 
                                         Concurrency::array_view<int> &csrRowPtrC, 
                                         Concurrency::array_view<int> &csrRowPtrCt, 
                                         Concurrency::array_view<int> &csrColIndCt, 
                                         Concurrency::array_view<T> &csrValCt, 
                                         int *_nnzCt, int m, int *_h_queue_one, 
                                         const hcsparseControl* control)
{
    //cl::Kernel kernel1  = KernelCache::get(control->queue,"SpGEMM_EM_kernels", "EM_mergepath", params);
    //cl::Kernel kernel2  = KernelCache::get(control->queue,"SpGEMM_EM_kernels", "EM_mergepath_global", params);
    
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;
    
       //int mergebuffer_size_local = 2304;
    
       //kWrapper2 << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC
         //                 << csrRowPtrCt << *csrColIndCt <<  *csrValCt << cl::__local((mergebuffer_size_local) * sizeof(int)) << cl::__local((mergebuffer_size_local) * sizeof(float)) << cl::__local(( num_threads+1) * sizeof(short)) << position << mergebuffer_size_local << cl::__local(sizeof(cl_int)   * (num_threads + 1)) << cl::__local(sizeof(cl_int)   * (num_threads + 1));


/*    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
    });
*/
    int temp_queue [6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;

    for (int i = position; i < position + num_blocks; i++)
    {
        if (queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = queue_one[TUPLE_QUEUE * i]; // row id
            if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
            {
                int accum = 0;
                switch (mergebuffer_size)
                {
                case 256:
                    accum = 512;
                    break;
                case 512:
                    accum = 1024;
                    break;
                case 1024:
                    accum = 2048;
                    break;
                case 2048:
                    accum = 2304;
                    break;
                case 2304:
                    accum = 2 * (2304 * 2);
                    break;
                }

                temp_queue[1] = *_nnzCt + counter * accum; // new start address
            }
            else if (mergepath_location == MERGEPATH_GLOBAL)
                temp_queue[1] = *_nnzCt + counter * (2 * (mergebuffer_size + 2304)); 
            temp_queue[2] = queue_one[TUPLE_QUEUE * i + 2]; // merged size
            temp_queue[3] = queue_one[TUPLE_QUEUE * i + 3]; // i
            temp_queue[4] = queue_one[TUPLE_QUEUE * i + 4]; // k
            temp_queue[5] = queue_one[TUPLE_QUEUE * i + 1]; // old start address

            queue_one[TUPLE_QUEUE * i]     = queue_one[TUPLE_QUEUE * (position + counter)];     // row id
            queue_one[TUPLE_QUEUE * i + 1] = queue_one[TUPLE_QUEUE * (position + counter) + 1]; // new start address
            queue_one[TUPLE_QUEUE * i + 2] = queue_one[TUPLE_QUEUE * (position + counter) + 2]; // merged size
            queue_one[TUPLE_QUEUE * i + 3] = queue_one[TUPLE_QUEUE * (position + counter) + 3]; // i
            queue_one[TUPLE_QUEUE * i + 4] = queue_one[TUPLE_QUEUE * (position + counter) + 4]; // k
            queue_one[TUPLE_QUEUE * i + 5] = queue_one[TUPLE_QUEUE * (position + counter) + 5]; // old start address

            queue_one[TUPLE_QUEUE * (position + counter)]     = temp_queue[0]; // row id
            queue_one[TUPLE_QUEUE * (position + counter) + 1] = temp_queue[1]; // new start address
            queue_one[TUPLE_QUEUE * (position + counter) + 2] = temp_queue[2]; // merged size
            queue_one[TUPLE_QUEUE * (position + counter) + 3] = temp_queue[3]; // i
            queue_one[TUPLE_QUEUE * (position + counter) + 4] = temp_queue[4]; // k
            queue_one[TUPLE_QUEUE * (position + counter) + 5] = temp_queue[5]; // old start address

            counter++;
            temp_num += queue_one[TUPLE_QUEUE * i + 2];
        }
    }

    if (counter > 0)
    {
        int nnzCt_new = 0;
        if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
        {
            int accum = 0;
            switch (mergebuffer_size)
            {
            case 256:
                accum = 512;
                break;
            case 512:
                accum = 1024;
                break;
            case 1024:
                accum = 2048;
                break;
            case 2048:
                accum = 2304;
                break;
            case 2304:
                accum = 2 * (2304 * 2);
                break;
            }

            nnzCt_new = *_nnzCt + counter * accum;
        }
        else if (mergepath_location == MERGEPATH_GLOBAL)
        nnzCt_new = *_nnzCt + counter * (2 * (mergebuffer_size + 2304));

        *_nnzCt = nnzCt_new;
    }

    *count_next = counter;

    return hcsparseSuccess;
}
 
template <typename T>
hcsparseStatus compute_nnzC_Ct_general(int *_h_counter_one, 
                                       Concurrency::array_view<int> &queue_one, 
                                       Concurrency::array_view<int> &csrRowPtrA, 
                                       Concurrency::array_view<int> &csrColIndA, 
                                       Concurrency::array_view<T> &csrValA, 
                                       Concurrency::array_view<int> &csrRowPtrB, 
                                       Concurrency::array_view<int> &csrColIndB, 
                                       Concurrency::array_view<T> &csrValB, 
                                       Concurrency::array_view<int> &csrRowPtrC, 
                                       Concurrency::array_view<int> &csrRowPtrCt, 
                                       Concurrency::array_view<int> &csrColIndCt, 
                                       Concurrency::array_view<T> &csrValCt, 
                                       int _n, int _nnzCt, int m, int *queue_one_h, 
                                       const hcsparseControl* control)
{
    int counter = 0;
    
    hcsparseStatus run_status;
    
    for (int j = 0; j < NUM_SEGMENTS; j++)
    {
        counter = _h_counter_one[j+1] - _h_counter_one[j];
        if (counter != 0)
        {

            if (j == 0)
            {
                int num_blocks = Concurrency::fast_math::ceil((double)counter / (double)GROUPSIZE_256);

                run_status = compute_nnzC_Ct_0<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrC, control);
            }
            else if (j == 1)
            {
                int num_blocks = Concurrency::fast_math::ceil((double)counter / (double)GROUPSIZE_256);

                run_status = compute_nnzC_Ct_1<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, control);
            }
            else if (j > 1 && j <= 32)
            {
                int num_blocks = Concurrency::fast_math::ceil((double)counter / (double)GROUPSIZE_256);
                run_status = compute_nnzC_Ct_2heap_noncoalesced_local<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, control);
            }
            else if (j > 32 && j <= 124)
            {
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan<T> (num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, _n, control);
            }
            else if (j == 127)
            {
                int count_next = counter;
                int num_threads, num_blocks, mergebuffer_size = 0;

                int num_threads_queue [5] = {64, 128, 256, 256, 256};
                int mergebuffer_size_queue [5] = {256, 512, 1024, 2048, 2304};

                while (count_next > 0)
                {
                    num_blocks = count_next;

                    num_threads = num_threads_queue[4];
                    mergebuffer_size += mergebuffer_size_queue[4];
                      
                    run_status = compute_nnzC_Ct_mergepath<T> (num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_GLOBAL, queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, &_nnzCt, m, queue_one_h, control);

                }

            }
      
        if (run_status != hcsparseSuccess)
            {
               return hcsparseInvalid;
            }
        }
    }
    
    return hcsparseSuccess;

}

template <typename T>
hcsparseStatus copy_Ct_to_C_Single(int num_blocks, int position, int size,
                                   Concurrency::array_view<T> &csrValC, 
                                   Concurrency::array_view<int> &csrRowPtrC, 
                                   Concurrency::array_view<int> &csrColIndC, 
                                   Concurrency::array_view<T> &csrValCt, 
                                   Concurrency::array_view<int> &csrRowPtrCt, 
                                   Concurrency::array_view<int> &csrColIndCt, 
                                   Concurrency::array_view<int> &queue_one, 
                                   const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int global_id = tidx.global[0];
        bool valid = (global_id < size);
        int row_id = valid ? queue_one[TUPLE_QUEUE * (position + global_id)] : 0;
        int Ct_base_start = valid ? queue_one[TUPLE_QUEUE * (position + global_id) + 1] : 0;
        int C_base_start  = valid ? csrRowPtrC[row_id] : 0;
        int colC   = valid ? csrColIndCt[Ct_base_start] : 0;
        T valC = valid ? csrValCt[Ct_base_start] : 0.0f;
        if (valid)
        {
            csrColIndC[C_base_start] = colC;
            csrValC[C_base_start]    = valC;
        }
    });

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus copy_Ct_to_C_Loopless(int num_blocks, int position, 
                                     Concurrency::array_view<T> &csrValC, 
                                     Concurrency::array_view<int> &csrRowPtrC, 
                                     Concurrency::array_view<int> &csrColIndC, 
                                     Concurrency::array_view<T> &csrValCt, 
                                     Concurrency::array_view<int> &csrRowPtrCt, 
                                     Concurrency::array_view<int> &csrColIndCt, 
                                     Concurrency::array_view<int> &queue_one, 
                                     const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int row_id = queue_one[TUPLE_QUEUE * (position + group_id)];
        int Ct_base_start = queue_one[TUPLE_QUEUE * (position + group_id) + 1] + local_id;
        int C_base_start  = csrRowPtrC[row_id]  + local_id;
        int C_base_stop   = csrRowPtrC[row_id + 1];
        if (C_base_start < C_base_stop)
        {
            csrColIndC[C_base_start] = csrColIndCt[Ct_base_start];
            csrValC[C_base_start]    = csrValCt[Ct_base_start];
        }
    });

    return hcsparseSuccess;
    
}

template <typename T>
hcsparseStatus copy_Ct_to_C_Loop(int num_blocks, int position, 
                                 Concurrency::array_view<T> &csrValC, 
                                 Concurrency::array_view<int> &csrRowPtrC, 
                                 Concurrency::array_view<int> &csrColIndC, 
                                 Concurrency::array_view<T> &csrValCt, 
                                 Concurrency::array_view<int> &csrRowPtrCt, 
                                 Concurrency::array_view<int> &csrColIndCt, 
                                 Concurrency::array_view<int> &queue_one, 
                                 const hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    Concurrency::extent<1> grdExt(szGlobalWorkSize);
    Concurrency::tiled_extent<GROUPSIZE_256> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<GROUPSIZE_256> tidx) restrict(amp)
    {
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int local_size = tidx.tile_dim0;
        int row_id = queue_one[TUPLE_QUEUE * (position + group_id)];
        int Ct_base_start = queue_one[TUPLE_QUEUE * (position + group_id) + 1];
        int C_base_start  = csrRowPtrC[row_id];
        int C_base_stop   = csrRowPtrC[row_id + 1];
        int stride        = C_base_stop - C_base_start;
        bool valid;
        int loop = ceil((float)stride / (float)local_size);
        C_base_start  += local_id;
        Ct_base_start += local_id;
        for (int i = 0; i < loop; i++)
        {
            valid = (C_base_start < C_base_stop);
            if (valid)
            {
                csrColIndC[C_base_start] = csrColIndCt[Ct_base_start];
                csrValC[C_base_start]    = csrValCt[Ct_base_start];
            }
            C_base_start += local_size;
            Ct_base_start += local_size;
        }
    });

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus copy_Ct_to_C_general(int *counter_one, 
                        Concurrency::array_view<T> &csrValC, 
                        Concurrency::array_view<int> &csrRowPtrC, 
                        Concurrency::array_view<int> &csrColIndC, 
                        Concurrency::array_view<T> &csrValCt, 
                        Concurrency::array_view<int> &csrRowPtrCt, 
                        Concurrency::array_view<int> &csrColIndCt, 
                        Concurrency::array_view<int> &queue_one, 
                        const hcsparseControl* control)
{
    int counter = 0;

    hcsparseStatus run_status;

    for (int j = 1; j < NUM_SEGMENTS; j++)
    {
        counter = counter_one[j+1] - counter_one[j];
        if (counter != 0)
        {
            if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks  = Concurrency::fast_math::ceil((double)counter / (double)num_threads);
                run_status = copy_Ct_to_C_Single<T> (num_blocks, counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            }
            else if (j > 1 && j <= 123)
                run_status = copy_Ct_to_C_Loopless<T> (counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 124)
                run_status = copy_Ct_to_C_Loop<T> (counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 127)
                run_status = copy_Ct_to_C_Loop<T> (counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
        }
    }
    
    return hcsparseSuccess;
}
 
template <typename T>
hcsparseStatus
csrSpGemm(const hcsparseCsrMatrix* matA,
          const hcsparseCsrMatrix* matB,
          hcsparseCsrMatrix* matC,
          const hcsparseControl* control )
{
    int m  = matA->num_rows;
    int k1 = matA->num_cols;
    int k2 = matB->num_rows;
    int n  = matB->num_cols;

    hcsparseStatus status1, status2;

    if(k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl; 
        return hcsparseInvalid;
    }  
    
    Concurrency::array_view<int> *csrRowPtrA = static_cast<Concurrency::array_view<int> *>(matA->rowOffsets);
    Concurrency::array_view<int> *csrColIndA = static_cast<Concurrency::array_view<int> *>(matA->colIndices);
    Concurrency::array_view<T> *csrValA = static_cast<Concurrency::array_view<T> *>(matA->values);
    Concurrency::array_view<int> *csrRowPtrB = static_cast<Concurrency::array_view<int> *>(matB->rowOffsets);
    Concurrency::array_view<int> *csrColIndB = static_cast<Concurrency::array_view<int> *>(matB->colIndices);
    Concurrency::array_view<T> *csrValB    = static_cast<Concurrency::array_view<T> *>(matB->values);
    
    Concurrency::array_view<int> *csrRowPtrC = static_cast<Concurrency::array_view<int> *>(matC->rowOffsets);

    int* csrRowPtrCt_h = (int*) calloc (m + 1, sizeof(int));
    Concurrency::array_view<int> csrRowPtrCt_d(m + 1, csrRowPtrCt_h);
 
    // STAGE 1
    compute_nnzCt<T> (m, *csrRowPtrA, *csrColIndA, *csrRowPtrB, *csrColIndB, csrRowPtrCt_d, control);
    
    // statistics
    int* counter = (int*) calloc (NUM_SEGMENTS, sizeof(int));
    int* counter_one = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* counter_sum = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* queue_one = (int*) calloc (m * TUPLE_QUEUE, sizeof(int));
    
    Concurrency::array_view<int> queue_one_d(m * TUPLE_QUEUE, queue_one); 

    // STAGE 2 - STEP 1 : statistics
    int nnzCt = statistics(csrRowPtrCt_h, counter, counter_one, counter_sum, queue_one, m);
    // STAGE 2 - STEP 2 : create Ct

    int* csrColIndCt_buf = (int*) calloc (nnzCt, sizeof(int));
    T* csrValCt_buf = (T*) calloc (nnzCt, sizeof(T));

    Concurrency::array_view<int> csrColIndCt(nnzCt, csrColIndCt_buf);
    Concurrency::array_view<T> csrValCt(nnzCt, csrValCt_buf);   
 
    // STAGE 3 - STEP 1 : compute nnzC and Ct
    status1 = compute_nnzC_Ct_general<T> (counter_one, queue_one_d, *csrRowPtrA, *csrColIndA, *csrValA, *csrRowPtrB, *csrColIndB, *csrValB, *csrRowPtrC, csrRowPtrCt_d, csrColIndCt, csrValCt, n, nnzCt, m, queue_one, control);

    int old_val, new_val;
    old_val = (*csrRowPtrC)[0];
    (*csrRowPtrC)[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = (*csrRowPtrC)[i];
        (*csrRowPtrC)[i] = old_val + (*csrRowPtrC)[i-1];
        old_val = new_val;
    }

    int nnzC = (*csrRowPtrC)[m];
    
    Concurrency::array_view<int> *csrColIndC = static_cast<Concurrency::array_view<int> *>(matC->colIndices);
    Concurrency::array_view<T> *csrValC = static_cast<Concurrency::array_view<T> *>(matC->values);

    status2 = copy_Ct_to_C_general<T> (counter_one, *csrValC, *csrRowPtrC, *csrColIndC, csrValCt, csrRowPtrCt_d, csrColIndCt, queue_one_d, control);
    
    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros  = nnzC;
 
    if (status1 == hcsparseSuccess && status2 == hcsparseSuccess)
        return hcsparseSuccess;
    else
        return hcsparseInvalid;
}

int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m)
{
    int nnzCt = 0;
    int _nnzCt_full = 0;

    // statistics for queues
    int count, position;

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            _h_counter_one[count]++;
            _h_counter_sum[count] += count;
            _nnzCt_full += count;
        }
        else if (count >= 122 && count <= 128)
        {
            _h_counter_one[122]++;
            _h_counter_sum[122] += count;
            _nnzCt_full += count;
        }
        else if (count >= 129 && count <= 256)
        {
            _h_counter_one[123]++;
            _h_counter_sum[123] += count;
            _nnzCt_full += count;
        }
        else if (count >= 257 && count <= 512)
        {
            _h_counter_one[124]++;
            _h_counter_sum[124] += count;
            _nnzCt_full += count;
        }
        else if (count >= 513)
        {
            _h_counter_one[127]++;
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _nnzCt_full += count;
        }
    }

    // exclusive scan

    int old_val, new_val;

    old_val = _h_counter_one[0];
    _h_counter_one[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_one[i];
        _h_counter_one[i] = old_val + _h_counter_one[i-1];
        old_val = new_val;
    }

    old_val = _h_counter_sum[0];
    _h_counter_sum[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_sum[i];
        _h_counter_sum[i] = old_val + _h_counter_sum[i-1];
        old_val = new_val;
    }

    nnzCt = _h_counter_sum[NUM_SEGMENTS];

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            position = _h_counter_one[count] + _h_counter[count];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[count];
            _h_counter_sum[count] += count;
            _h_counter[count]++;
        }
        else if (count >= 122 && count <= 128)
        {
            position = _h_counter_one[122] + _h_counter[122];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[122];
            _h_counter_sum[122] += count;
            _h_counter[122]++;
        }
        else if (count >= 129 && count <= 256)
        {
            position = _h_counter_one[123] + _h_counter[123];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[123];
            _h_counter_sum[123] += count;
            _h_counter[123]++;
        }
        else if (count >= 257 && count <= 512)
        {
            position = _h_counter_one[124] + _h_counter[124];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[124];
            _h_counter_sum[124] += count;
            _h_counter[124]++;
        }
        else if (count >= 513)
        {
            position = _h_counter_one[127] + _h_counter[127];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[127];
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _h_counter[127]++;
        }
    }

    return nnzCt;
}
