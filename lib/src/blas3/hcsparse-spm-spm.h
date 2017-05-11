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
#define mergebuffer_size_local 2304
 
int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m);

template <typename T>
inline
void binarysearch (int   *s_key,
                   T     *s_val,
                   int   key_input,
                   T     val_input,
                   int   merged_size,
                   bool  *is_new_col) [[hc]]
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];
        
        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] += val_input;
            *is_new_col = 0;
            break;
        }
    }
}

template <typename T>
inline
void binarysearch_sub (int   *s_key,
                       T     *s_val,
                       int   key_input,
                       T     val_input,
                       int   merged_size) [[hc]]
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = s_key[median];
        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            s_val[median] -= val_input;
            break;
        }
    }
}

template <typename T>
inline
void binarysearch_global (int *d_key,
                          T   *d_val,
                          int                 key_input,
                          T                   val_input,
                          int                 merged_size,
                          bool                *is_new_col) [[hc]]
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_key[median];
        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            d_val[median] += val_input;
            *is_new_col = 0;
            break;
        }
    }
}

template <typename T>
inline
void binarysearch_global_sub (int *d_key,
                              T   *d_val,
                              int                 key_input,
                              T                   val_input,
                              int                 merged_size) [[hc]]
{
    int start = 0;
    int stop  = merged_size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_key[median];
        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else
        {
            // atomicAdd is not needed since duplicate is not existed in each input row
            d_val[median] -= val_input;
            break;
        }
    }
}

inline
void scan_256 (hc::tiled_index<1> &tidx,
               int                *s_scan,
               const int          local_id) [[hc]]
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    int temp;
    if (local_id < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    tidx.barrier.wait();
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    tidx.barrier.wait();
    if (local_id < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

inline
int y_pos (const int x_pos,
           const int b_length,
           const int offset) [[hc]]
{
    int pos = b_length - (x_pos + b_length - offset);
    return pos > b_length ? b_length : pos;
}

inline
int mergepath_partition_liu (int            *s_a_key,
                             const int      a_length,
                             int            *s_b_key,
                             const int      b_length,
                             const int      offset) [[hc]]
{
    int x_start = offset > b_length ? offset - b_length : 0;
    int x_stop  = offset > a_length ? a_length : offset;
    
    int x_median;
    
    while (x_stop >= x_start)
    {
        x_median = (x_stop + x_start) / 2;
        
        if (s_a_key[x_median] > s_b_key[y_pos(x_median, b_length, offset) - 1])
        {
            if (s_a_key[x_median - 1] < s_b_key[y_pos(x_median, b_length, offset)])
            {
                break;
            }
            else
            {
                x_stop = x_median - 1;
            }
        }
        else
        {
            x_start = x_median + 1;
        }
    }
    
    return x_median;
}

template <typename T>
inline
void mergepath_serialmerge_liu (int         *s_a_key,
                                T           *s_a_val,
                                const int   a_length,
                                int         *s_b_key,
                                T           *s_b_val,
                                const int   b_length,
                                int         *reg_key,
                                T           *reg_val) [[hc]]
{
    int a_pointer = 0;
    int b_pointer = 0;
    
    for (int c_pointer = 0; c_pointer < a_length + b_length; c_pointer++)
    {
        if (a_pointer < a_length && (b_pointer >= b_length || s_a_key[a_pointer] <= s_b_key[b_pointer]))
        {
            reg_key[c_pointer] = s_a_key[a_pointer];
            reg_val[c_pointer] = s_a_val[a_pointer];
            a_pointer += 1;
        }
        else
        {
            reg_key[c_pointer] = s_b_key[b_pointer];
            reg_val[c_pointer] = s_b_val[b_pointer];
            b_pointer += 1;
        }
    }
}

template <typename T>
inline
void mergepath_liu (hc::tiled_index<1> &tidx,
                    int                *s_a_key,
                    T                  *s_a_val,
                    const int          a_length,
                    int                *s_b_key,
                    T                  *s_b_val,
                    const int          b_length,
                    int                *s_a_border,
                    int                *s_b_border,
                    int                *reg_key,
                    T                  *reg_val) [[hc]]
{
    if (b_length == 0)
        return;
    if (s_a_key[a_length-1] < s_b_key[0])
        return;
    int local_id = tidx.local[0];
    int local_size = tidx.tile_dim[0];
    int delta = hc::fast_math::ceil((float)(a_length + b_length) / (float)local_size);
    int active_threads = hc::fast_math::ceil((float)(a_length + b_length) / (float)delta);
    int offset = delta * local_id;
    int a_start, a_stop, b_start, b_stop;
    if (!local_id)
    {
        s_a_border[active_threads] = a_length;
        s_b_border[active_threads] = b_length;
    }
    if (local_id < active_threads)
    {
        s_a_border[local_id] = a_start = mergepath_partition_liu(s_a_key, a_length, s_b_key, b_length, offset);
        s_b_border[local_id] = b_start = y_pos(s_a_border[local_id], b_length, offset);
    }
    tidx.barrier.wait();
    if (local_id < active_threads)
    {
        a_stop = s_a_border[local_id+1];
        b_stop = s_b_border[local_id+1];
    }
    if (local_id < active_threads)
    {
        mergepath_serialmerge_liu<T> (&s_a_key[a_start],
                                      &s_a_val[a_start],
                                      a_stop - a_start,
                                      &s_b_key[b_start],
                                      &s_b_val[b_start],
                                      b_stop - b_start,
                                      reg_key, reg_val);
    }
    tidx.barrier.wait();
    if (local_id < active_threads)
    {
        for (int is = 0; is < (a_stop - a_start) + (b_stop - b_start); is++)
        {
            s_a_key[offset + is] = reg_key[is];
            s_a_val[offset + is] = reg_val[is];
        }
    }
    tidx.barrier.wait();
}

inline
int mergepath_partition_global_liu (int *s_a_key,
                                    const int           a_length,
                                    int *s_b_key,
                                    const int           b_length,
                                    const int           offset) [[hc]]
{
    int x_start = offset > b_length ? offset - b_length : 0;
    int x_stop  = offset > a_length ? a_length : offset;
    int x_median;
    while (x_stop >= x_start)
    {
        x_median = (x_stop + x_start) / 2;
        
        if (s_a_key[x_median] > s_b_key[y_pos(x_median, b_length, offset) - 1])
        {
            if (s_a_key[x_median - 1] < s_b_key[y_pos(x_median, b_length, offset)])
            {
                break;
            }
            else
            {
                x_stop = x_median - 1;
            }
        }
        else
        {
            x_start = x_median + 1;
        }
    }
    return x_median;
}

template <typename T>
inline
void mergepath_global_2level_liu (hc::tiled_index<1>  &tidx,
                                  int *s_a_key,
                                  T   *s_a_val,
                                  const int           a_length,
                                  int *s_b_key,
                                  T   *s_b_val,
                                  const int           b_length,
                                  int                 *s_a_border,
                                  int                 *s_b_border,
                                  int                 *reg_key,
                                  T                   *reg_val,
                                  int                 *s_key,
                                  T                   *s_val,
                                  int *d_temp_key,
                                  T   *d_temp_val) [[hc]]
{
    if (b_length == 0)
        return;
    if (s_a_key[a_length-1] < s_b_key[0])
        return;

    int local_id = tidx.local[0];
    int local_size = tidx.tile_dim[0];
    int delta_2level = local_size * 9;
    int loop_2level = hc::fast_math::ceil((float)(a_length + b_length) / (float)delta_2level);
    int a_border_2level_l, b_border_2level_l, a_border_2level_r, b_border_2level_r;
    for (int i = 0; i < loop_2level; i++)
    {
        // compute `big' borders
        int offset_2level = delta_2level * i;
        a_border_2level_l = i == 0 ? 0 : a_border_2level_r;
        b_border_2level_l = i == 0 ? 0 : b_border_2level_r;
        int offset_2level_next = delta_2level * (i + 1);
        if (i == (loop_2level - 1)){
            a_border_2level_r = a_length;
            b_border_2level_r = b_length;
        }
        else
        {
            s_a_border[local_id] = a_border_2level_r = local_id < 64 ? mergepath_partition_global_liu(s_a_key, a_length, s_b_key, b_length, offset_2level_next) : 0;
            tidx.barrier.wait();
            a_border_2level_r = local_id < 64 ? a_border_2level_r : s_a_border[local_id % 64];
            b_border_2level_r = y_pos(a_border_2level_r, b_length, offset_2level_next);
        }
        //barrier(CLK_GLOBAL_MEM_FENCE);
        // load entries in the borders
        int a_size = a_border_2level_r - a_border_2level_l;
        int b_size = b_border_2level_r - b_border_2level_l;
        for (int j = local_id; j < a_size; j += local_size)
        {
            s_key[j] = s_a_key[a_border_2level_l + j];
            s_val[j] = s_a_val[a_border_2level_l + j];
        }
        for (int j = local_id; j < b_size; j += local_size)
        {
            s_key[a_size + j] = s_b_key[b_border_2level_l + j];
            s_val[a_size + j] = s_b_val[b_border_2level_l + j];
        }
        tidx.barrier.wait();
        // merge path in local mem
        mergepath_liu<T> (tidx, s_key, s_val, a_size,
                          &s_key[a_size], &s_val[a_size], b_size,
                          s_a_border, s_b_border, reg_key, reg_val);
        tidx.barrier.wait();
        // dump the merged part to device mem (temp)
        for (int j = local_id; j < a_size + b_size; j += local_size)
        {
            d_temp_key[offset_2level + j] = s_key[j];
            d_temp_val[offset_2level + j] = s_val[j];
        }
        tidx.barrier.wait();
    }
    // dump the temp data to the target place, both in device mem
    for (int j = local_id; j < a_length + b_length; j += local_size)
    {
        s_a_key[j] = d_temp_key[j];
        s_a_val[j] = d_temp_val[j];
    }
    tidx.barrier.wait();
}

template <typename T>
inline
void readwrite_mergedlist_global (hc::tiled_index<1>        &tidx,
                                  int *d_csrColIndCt,
                                  T   *d_csrValCt,
                                  int       *d_key_merged,
                                  T         *d_val_merged,
                                  const int                 merged_size,
                                  const int                 row_offset,
                                  const bool                is_write) [[hc]]
{
    int local_id = tidx.local[0];
    int local_size = tidx.tile_dim[0];
    int stride, offset_local_id, global_offset;
    int loop = hc::fast_math::ceil((float)merged_size / (float)local_size);
    for (int i = 0; i < loop; i++)
    {
        stride = i != loop - 1 ? local_size : merged_size - i * local_size;
        offset_local_id = i * local_size + local_id;
        global_offset = row_offset + offset_local_id;
        if (local_id < stride)
        {
            if (is_write)
            {
                d_csrColIndCt[global_offset] = d_key_merged[offset_local_id];
                d_csrValCt[global_offset]    = d_val_merged[offset_local_id];
            }
            else
            {
                d_key_merged[offset_local_id] = d_csrColIndCt[global_offset];
                d_val_merged[offset_local_id] = d_csrValCt[global_offset];
            }
        }
    }
}

template <typename T>
inline
void readwrite_mergedlist (hc::tiled_index<1>  &tidx, 
                           int *d_csrColIndCt,
                           T   *d_csrValCt,
                           int                 *s_key_merged,
                           T                   *s_val_merged,
                           const int           merged_size,
                           const int           row_offset,
                           const bool          is_write) [[hc]]
{
    int local_id = tidx.local[0];
    int local_size = tidx.tile_dim[0];
    int stride, offset_local_id, global_offset;
    int loop = hc::fast_math::ceil((float)merged_size / (float)local_size);
    for (int i = 0; i < loop; i++)
    {
        stride = i != loop - 1 ? local_size : merged_size - i * local_size;
        offset_local_id = i * local_size + local_id;
        global_offset = row_offset + offset_local_id;
        if (local_id < stride)
        {
            if (is_write)
            {
                d_csrColIndCt[global_offset] = s_key_merged[offset_local_id];
                d_csrValCt[global_offset]    = s_val_merged[offset_local_id];
            }
            else
            {
                s_key_merged[offset_local_id] = d_csrColIndCt[global_offset];
                s_val_merged[offset_local_id] = d_csrValCt[global_offset];
            }
        }
    }
}

template <typename T>
inline
void siftDown (int       *s_key,
               T         *s_val,
               const int start,
               const int stop,
               const int local_id,
               const int local_size) [[hc]] 
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
int heapsort (int       *s_key,
              T         *s_val,
              const int segment_size,
              const int local_id,
              const int local_size) [[hc]]
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
void coex (int        *keyA,
           T          *valA,
           int        *keyB,
           T          *valB,
           const int  dir) [[hc]]
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
void bitonic (int   *s_key,
              T     *s_val,
              int   stage,
              int   passOfStage,
              int   local_id,
              int   local_counter) [[hc]]
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
void bitonicsort (hc::tiled_index<1> &tidx,
                  int                *s_key,
                  T                  *s_val,
                  int                arrayLength) [[hc]]
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

template <typename T>
inline
void compression_scan (hc::tiled_index<1> &tidx,
                       int                *s_scan,
                       int                *s_key,
                       T                  *s_val,
                       const int          local_counter,
                       const int          local_size,
                       const int          local_id,
                       const int          local_id_halfwidth) [[hc]]
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
hcsparseStatus compute_nnzCt (int m, 
                              int *csrRowPtrA, 
                              int *csrColIndA, 
                              int *csrRowPtrB, 
                              int *csrColIndB, 
                              int *csrRowPtrCt, 
                              hcsparseControl* control)
{  
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    int num_threads = GROUPSIZE_256;
    int num_blocks = hc::fast_math::ceil((double)m / (double)num_threads);

    szLocalWorkSize  = num_threads;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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
    }).wait();

    return hcsparseSuccess;
 }
 

template <typename T> 
hcsparseStatus compute_nnzC_Ct_0 (int num_blocks, int j, 
                                  int counter, int position, 
                                  int *queue_one, 
                                  int *csrRowPtrC, 
                                  hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;
    
    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        int global_id = tidx.global[0];
        if (global_id < counter)
        {
            int row_id = queue_one[TUPLE_QUEUE * (position + global_id)];
            csrRowPtrC[row_id] = 0;
        }
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_1 (int num_blocks, int j, 
                                  int counter, int position, 
                                  int *queue_one, 
                                  int *csrRowPtrA, 
                                  int *csrColIndA, 
                                  T *csrValA, 
                                  int *csrRowPtrB, 
                                  int *csrColIndB, 
                                  T *csrValB, 
                                  int *csrRowPtrC, 
                                  int *csrRowPtrCt, 
                                  int *csrColIndCt, 
                                  T *csrValCt, 
                                  hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> tidx) [[hc]]
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
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_2heap_noncoalesced_local (int num_blocks, int j, 
                                                         int counter, int position, 
                                                         int *queue_one, 
                                                         int *csrRowPtrA, 
                                                         int *csrColIndA, 
                                                         T *csrValA, 
                                                         int *csrRowPtrB, 
                                                         int *csrColIndB, 
                                                         T *csrValB, 
                                                         int *csrRowPtrC, 
                                                         int *csrRowPtrCt, 
                                                         int *csrColIndCt, 
                                                         T *csrValCt, 
                                                         hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

std::cout << "num_blocks = " << num_blocks << " counter = " << counter << std::endl;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static int s_key[GROUPSIZE_256 * 32]; // NUM_THREADS * MAX SEGMENT PROCESSED
        tile_static T s_val[GROUPSIZE_256 * 32];
        const int local_id = tidx.local[0];
        const int group_id = tidx.tile[0];
        const int global_id = tidx.global[0];
        const int local_size = tidx.tile_dim[0];
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
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_bitonic_scan (int num_blocks, int j, int position, 
                                             int *queue_one, 
                                             int *csrRowPtrA, 
                                             int *csrColIndA, 
                                             T *csrValA, 
                                             int *csrRowPtrB, 
                                             int *csrColIndB, 
                                             T *csrValB, 
                                             int *csrRowPtrC, 
                                             int *csrRowPtrCt, 
                                             int *csrColIndCt, 
                                             T *csrValCt, 
                                             int n, 
                                             hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static int s_key[2*GROUPSIZE_256];
        tile_static T s_val[2*GROUPSIZE_256];
        tile_static int s_scan[2*GROUPSIZE_256 + 1];
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int local_size = tidx.tile_dim[0];
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
    }).wait();
    
    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus compute_nnzC_Ct_mergepath (int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location, 
                                          int *queue_one, 
                                          int *csrRowPtrA, 
                                          int *csrColIndA, 
                                          T *csrValA, 
                                          int *csrRowPtrB, 
                                          int *csrColIndB, 
                                          T *csrValB, 
                                          int *csrRowPtrC, 
                                          int *csrRowPtrCt, 
                                          int *csrColIndCt, 
                                          T *csrValCt, 
                                          int *_nnzCt, int m, int *_h_queue_one, 
                                          hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;
    
    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static int s_key_merged_l1[mergebuffer_size_local];
        tile_static T s_val_merged_l1[mergebuffer_size_local];
        tile_static int s_scan[GROUPSIZE_256+1];
        tile_static int s_a_border[GROUPSIZE_256+1];
        tile_static int s_b_border[GROUPSIZE_256+1];

        int queue_id = TUPLE_QUEUE * (position + tidx.tile[0]);
        // if merged size equals -1, kernel return since this row is done
        int merged_size_l2 = queue_one[queue_id + 2];
        int merged_size_l1 = 0;
        int local_id = tidx.local[0]; //threadIdx.x;
        int row_id = queue_one[queue_id];
        int   local_size = tidx.tile_dim[0];
        float local_size_float = local_size;
        int stride, loop;
        int reg_reuse1;
        int   col_Ct;      // index_type
        T val_Ct;      // value_type
        T val_A;       // value_type
        int start_col_index_A, stop_col_index_A;  // index_type
        int start_col_index_B, stop_col_index_B;  // index_type
        int k;
        bool  is_new_col;
        bool  is_last;
        int   reg_key[9];
        T reg_val[9];
        start_col_index_A = csrRowPtrA[row_id];
        stop_col_index_A  = csrRowPtrA[row_id + 1];
        is_last = true;
        start_col_index_A = queue_one[queue_id + 3];
        // load existing merged list
        reg_reuse1 = queue_one[queue_id + 1];

        int *d_key_merged = &csrColIndCt[reg_reuse1];
        T *d_val_merged = &csrValCt[reg_reuse1];
        reg_reuse1 = queue_one[queue_id + 5];
        readwrite_mergedlist_global<T> (tidx, csrColIndCt, csrValCt, d_key_merged, d_val_merged, merged_size_l2, reg_reuse1, 0);
        tidx.barrier.wait();
        // merge the rest of sets of current nnzCt row to the merged list
        while (start_col_index_A < stop_col_index_A)
        {
            reg_reuse1 = csrColIndA[start_col_index_A];                      // reg_reuse1 = row_id_B
	    val_A    = csrValA[start_col_index_A];
            start_col_index_B = is_last ? queue_one[queue_id + 4] : csrRowPtrB[reg_reuse1];      // reg_reuse1 = row_id_B
            is_last = false;
            stop_col_index_B  = csrRowPtrB[reg_reuse1 + 1];  // reg_reuse1 = row_id_B
            stride = stop_col_index_B - start_col_index_B;
            loop  = hc::fast_math::ceil(stride / local_size_float); //ceil((float)stride / (float)local_size);
            start_col_index_B += local_id;
            for (k = 0; k < loop; k++)
            {
                tidx.barrier.wait();
                is_new_col = 0;
                if (start_col_index_B < stop_col_index_B)
                {
                    col_Ct = csrColIndB[start_col_index_B];
                    val_Ct = csrValB[start_col_index_B] * val_A;
                    // binary search on existing sorted list
                    // if the column is existed, add the value to the position
                    // else, set scan value to 1, and wait for scan
                    is_new_col = 1;
                    // search on l2
                    binarysearch_global<T> (d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2, &is_new_col);
                    // search on l1
                    if (is_new_col == 1)
                        binarysearch<T>(s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1, &is_new_col);
                }
                s_scan[local_id] = is_new_col;
                tidx.barrier.wait();
                // scan with half-local_size work-items
                // s_scan[local_size] is the size of input non-duplicate array
                scan_256(tidx, s_scan, local_id);
                tidx.barrier.wait();
                // if all elements are absorbed into merged list,
                // the following work in this inner-loop is not needed any more
                if (s_scan[local_size] == 0)
                {
                    start_col_index_B += local_size;
                    continue;
                }
                // check if the total size is larger than the capicity of merged list
                if (merged_size_l1 + s_scan[local_size] > mergebuffer_size)
                {
                    if (start_col_index_B < stop_col_index_B)
                    {
                        // rollback on l2
                        binarysearch_global_sub<T> (d_key_merged, d_val_merged, col_Ct, val_Ct, merged_size_l2);
                        // rollback on l1
                        binarysearch_sub<T> (s_key_merged_l1, s_val_merged_l1, col_Ct, val_Ct, merged_size_l1);
                    }
                    tidx.barrier.wait();
                    // write a signal to some place, not equals -1 means next round is needed
                    if (local_id == 0)
                    {
                        queue_one[queue_id + 2] = merged_size_l2 + merged_size_l1;
                        queue_one[queue_id + 3] = start_col_index_A;
                        queue_one[queue_id + 4] = start_col_index_B;
                    }
                    // dump l1 to global
                    readwrite_mergedlist<T> (tidx, d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                                             merged_size_l1, merged_size_l2, 1);
                    tidx.barrier.wait();
                
                    mergepath_global_2level_liu<T> (tidx, d_key_merged, d_val_merged, merged_size_l2,
                                                    &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                                                    s_a_border, s_b_border,
                                                    reg_key, reg_val,
                                                    s_key_merged_l1, s_val_merged_l1,
                                                    &d_key_merged[merged_size_l2 + merged_size_l1],
                                                    &d_val_merged[merged_size_l2 + merged_size_l1]);
                    return;
                }
                // write compact input to free place in merged list
                if(is_new_col)
                {
                    reg_reuse1 = merged_size_l1 + s_scan[local_id];
                    s_key_merged_l1[reg_reuse1] = col_Ct;
                    s_val_merged_l1[reg_reuse1] = val_Ct;
                }
                tidx.barrier.wait();
                // merge path partition on l1
                reg_reuse1 = s_scan[local_size]; // reg_reuse1 = size_b;
                mergepath_liu<T> (tidx, s_key_merged_l1, s_val_merged_l1, merged_size_l1, 
                                  &s_key_merged_l1[merged_size_l1], &s_val_merged_l1[merged_size_l1], reg_reuse1,
                                  s_a_border, s_b_border, reg_key, reg_val);
                merged_size_l1 += reg_reuse1; // reg_reuse1 = size_b = s_scan[local_size];
                start_col_index_B += local_size;
            }
            start_col_index_A++;
        }
        tidx.barrier.wait();
        if (local_id == 0)
        {
            csrRowPtrC[row_id] = merged_size_l2 + merged_size_l1;
            queue_one[queue_id + 2] = -1;
        }
        // dump l1 to global
        readwrite_mergedlist<T> (tidx, d_key_merged, d_val_merged, s_key_merged_l1, s_val_merged_l1,
                                 merged_size_l1, merged_size_l2, 1);
        tidx.barrier.wait(); 
    
        mergepath_global_2level_liu<T> (tidx, d_key_merged, d_val_merged, merged_size_l2,
                                        &d_key_merged[merged_size_l2], &d_val_merged[merged_size_l2], merged_size_l1,
                                        s_a_border, s_b_border,
                                        reg_key, reg_val,
                                        s_key_merged_l1, s_val_merged_l1,
                                        &d_key_merged[merged_size_l2 + merged_size_l1],
                                        &d_val_merged[merged_size_l2 + merged_size_l1]);
    }).wait();

    control->accl_view.copy(queue_one, _h_queue_one, m * TUPLE_QUEUE * sizeof(int)); 

    int temp_queue [6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;

    for (int i = position; i < position + num_blocks; i++)
    {
        if (queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = queue_one[TUPLE_QUEUE * i]; // row id
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
 
    control->accl_view.copy(_h_queue_one, queue_one, m * TUPLE_QUEUE * sizeof(int)); 

    if (counter > 0)
    {
        int nnzCt_new = 0;
        nnzCt_new = *_nnzCt + counter * (2 * (mergebuffer_size + 2304));

        *_nnzCt = nnzCt_new;
    }

    *count_next = counter;

    return hcsparseSuccess;
}
 
template <typename T>
hcsparseStatus compute_nnzC_Ct_general (int *_h_counter_one, 
                                        int *queue_one,
                                        int *csrRowPtrA, 
                                        int *csrColIndA, 
                                        T *csrValA, 
                                        int *csrRowPtrB, 
                                        int *csrColIndB, 
                                        T *csrValB, 
                                        int *csrRowPtrC, 
                                        int *csrRowPtrCt, 
                                        int *csrColIndCt, 
                                        T *csrValCt, 
                                        int _n, int _nnzCt, int m, int *queue_one_h, 
                                        hcsparseControl* control)
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
                int num_blocks = hc::fast_math::ceil((double)counter / (double)GROUPSIZE_256);

                run_status = compute_nnzC_Ct_0<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrC, control);
            }
            else if (j == 1)
            {
                int num_blocks = hc::fast_math::ceil((double)counter / (double)GROUPSIZE_256);

                run_status = compute_nnzC_Ct_1<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB,
                                                   csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, control);
            }
            else if (j > 1 && j <= 32)
            {
                int num_blocks = hc::fast_math::ceil((double)counter / (double)GROUPSIZE_256);

                for (int i = 0 ; i  < counter; i++)
                   std::cout << "queue[" << TUPLE_QUEUE * (_h_counter_one[j] + i) << "] = " << queue_one_h \
                               [TUPLE_QUEUE * (_h_counter_one[j] + i)] <<std::endl;
 
                run_status = compute_nnzC_Ct_2heap_noncoalesced_local<T> (num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA,
                                                                          csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, control);
            }
#if 0
            else if (j > 32 && j <= 124)
            {
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan<T> (num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB,
                                                              csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, _n, control);
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
                      
                    run_status = compute_nnzC_Ct_mergepath<T> (num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_GLOBAL, queue_one, csrRowPtrA, csrColIndA,
                                                               csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, &_nnzCt, m, queue_one_h, control);

                }

            }
#endif
      
            if (run_status != hcsparseSuccess)
            {
               return hcsparseInvalid;
            }
        }
    }
    
    return hcsparseSuccess;

}

template <typename T>
hcsparseStatus copy_Ct_to_C_Single (int num_blocks, int position, int size,
                                    T *csrValC, 
                                    int *csrRowPtrC, 
                                    int *csrColIndC, 
                                    T *csrValCt, 
                                    int *csrRowPtrCt, 
                                    int *csrColIndCt, 
                                    int *queue_one, 
                                    hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus copy_Ct_to_C_Loopless (int num_blocks, int position, 
                                      T *csrValC, 
                                      int *csrRowPtrC, 
                                      int *csrColIndC, 
                                      T *csrValCt, 
                                      int *csrRowPtrCt, 
                                      int *csrColIndCt, 
                                      int *queue_one, 
                                      hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize  = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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
    }).wait();

    return hcsparseSuccess;
    
}

template <typename T>
hcsparseStatus copy_Ct_to_C_Loop (int num_blocks, int position, 
                                  T *csrValC, 
                                  int *csrRowPtrC, 
                                  int *csrColIndC, 
                                  T *csrValCt, 
                                  int *csrRowPtrCt, 
                                  int *csrColIndCt, 
                                  int *queue_one, 
                                  hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;

    szLocalWorkSize = GROUPSIZE_256;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUPSIZE_256);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int local_size = tidx.tile_dim[0];
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
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus copy_Ct_to_C_general (int *counter_one, 
                                     T *csrValC, 
                                     int *csrRowPtrC, 
                                     int *csrColIndC, 
                                     T *csrValCt, 
                                     int *csrRowPtrCt, 
                                     int *csrColIndCt, 
                                     int *queue_one, 
                                     hcsparseControl* control)
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
                int num_blocks  = hc::fast_math::ceil((double)counter / (double)num_threads);
                run_status = copy_Ct_to_C_Single<T> (num_blocks, counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            }
#if 0
            else if (j > 1 && j <= 123)
                run_status = copy_Ct_to_C_Loopless<T> (counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
#endif
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
csrSpGemm(hcsparseControl* control,          
          int m,
          int n,
          int k,
          const float *csrValA,
          const int *csrRowPtrA,
          const int *csrColIndA,
          const float *csrValB,
          const int *csrRowPtrB,
          const int *csrColIndB,
          float *csrValC,
          const int *csrRowPtrC,
          int *csrColIndC)
{
    hcsparseStatus status1, status2;
    hc::accelerator acc = (control->accl_view).get_accelerator();

    int* csrRowPtrCt_h = (int*) calloc (m + 1, sizeof(int));
    int* csrRowPtrCt_d = (int*) am_alloc((m + 1) * sizeof(int), acc, 0);
 
    // STAGE 1
    compute_nnzCt<T> (m, (int *)csrRowPtrA, (int *)csrColIndA, (int *)csrRowPtrB, (int *)csrColIndB, csrRowPtrCt_d, control);

    control->accl_view.copy(csrRowPtrCt_d, csrRowPtrCt_h, (m + 1) * sizeof(int));   
 
    // statistics
    int* counter = (int*) calloc (NUM_SEGMENTS, sizeof(int));
    int* counter_one = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* counter_sum = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* queue_one = (int*) calloc (m * TUPLE_QUEUE, sizeof(int));
    
    // STAGE 2 - STEP 1 : statistics
    int nnzCt = statistics(csrRowPtrCt_h, counter, counter_one, counter_sum, queue_one, m);
    // STAGE 2 - STEP 2 : create Ct

    int *queue_one_d = (int*) am_alloc(m * TUPLE_QUEUE * sizeof(int), acc, 0);
    control->accl_view.copy(queue_one, queue_one_d, m * TUPLE_QUEUE * sizeof(int));

    int *csrColIndCt = (int*) am_alloc(nnzCt * sizeof(int), acc, 0);
    T *csrValCt = (T*) am_alloc(nnzCt * sizeof(T), acc, 0);

    // STAGE 3 - STEP 1 : compute nnzC and Ct
    status1 = compute_nnzC_Ct_general<T>
                   (counter_one, queue_one_d, (int *)csrRowPtrA, (int *)csrColIndA, 
                    (float *)csrValA, (int *)csrRowPtrB, (int *)csrColIndB, (float *)csrValB,
                    (int *)csrRowPtrC, csrRowPtrCt_d, csrColIndCt,
                    csrValCt, n, nnzCt, m, queue_one, control);

    int *csrRowPtrC_h = (int*) calloc(m + 1, sizeof(int));
    control->accl_view.copy(csrRowPtrC, csrRowPtrC_h, (m + 1) * sizeof(int));

    int old_val, new_val;
    old_val = csrRowPtrC_h[0];
    csrRowPtrC_h[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrC_h[i];
        csrRowPtrC_h[i] = old_val + csrRowPtrC_h[i-1];
        old_val = new_val;
    }

    int nnzC = csrRowPtrC_h[m];

    control->accl_view.copy(csrRowPtrC_h, (void *)csrRowPtrC, (m + 1) * sizeof(int));
    
    status2 = copy_Ct_to_C_general<T> (counter_one, csrValC, (int*)csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt_d, csrColIndCt, queue_one_d, control);
    
    if (status1 == hcsparseSuccess && status2 == hcsparseSuccess)
        return hcsparseSuccess;
    else
        return hcsparseInvalid;
} 

template <typename T>
hcsparseStatus
csrSpGemm (const hcsparseCsrMatrix* matA,
           const hcsparseCsrMatrix* matB,
           hcsparseCsrMatrix* matC,
           hcsparseControl* control )
{
    int m  = matA->num_rows;
    int k1 = matA->num_cols;
    int k2 = matB->num_rows;
    int n  = matB->num_cols;

    hc::accelerator acc = (control->accl_view).get_accelerator();

    hcsparseStatus status1, status2;

    if(k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl; 
        return hcsparseInvalid;
    }  
    
    int *csrRowPtrA = static_cast<int*>(matA->rowOffsets);
    int *csrColIndA = static_cast<int*>(matA->colIndices);
    T *csrValA = static_cast<T*>(matA->values);
    int *csrRowPtrB = static_cast<int*>(matB->rowOffsets);
    int *csrColIndB = static_cast<int*>(matB->colIndices);
    T *csrValB    = static_cast<T*>(matB->values);
    
    int *csrRowPtrC = static_cast<int*>(matC->rowOffsets);

    int* csrRowPtrCt_h = (int*) calloc (m + 1, sizeof(int));
    int* csrRowPtrCt_d = (int*) am_alloc((m + 1) * sizeof(int), acc, 0);
 
    // STAGE 1
    compute_nnzCt<T> (m, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, control);

    control->accl_view.copy(csrRowPtrCt_d, csrRowPtrCt_h, (m + 1) * sizeof(int));   
 
    // statistics
    int* counter = (int*) calloc (NUM_SEGMENTS, sizeof(int));
    int* counter_one = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* counter_sum = (int*) calloc (NUM_SEGMENTS + 1, sizeof(int));
    int* queue_one = (int*) calloc (m * TUPLE_QUEUE, sizeof(int));
    
    // STAGE 2 - STEP 1 : statistics
    int nnzCt = statistics(csrRowPtrCt_h, counter, counter_one, counter_sum, queue_one, m);
    // STAGE 2 - STEP 2 : create Ct

    int *queue_one_d = (int*) am_alloc(m * TUPLE_QUEUE * sizeof(int), acc, 0);
    control->accl_view.copy(queue_one, queue_one_d, m * TUPLE_QUEUE * sizeof(int));

    int *csrColIndCt = (int*) am_alloc(nnzCt * sizeof(int), acc, 0);
    T *csrValCt = (T*) am_alloc(nnzCt * sizeof(T), acc, 0);
 
    // STAGE 3 - STEP 1 : compute nnzC and Ct
    status1 = compute_nnzC_Ct_general<T> (counter_one, queue_one_d, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB,
                                          csrValB, csrRowPtrC, csrRowPtrCt_d, csrColIndCt, csrValCt, n, nnzCt, m, queue_one, control);

    int *csrRowPtrC_h = (int*) calloc(m + 1, sizeof(int));
    control->accl_view.copy(csrRowPtrC, csrRowPtrC_h, (m + 1) * sizeof(int));

    for (int i = 0 ;i < m + 1; i++)
      std::cout << "csrRowPtr[" << i << "] = " << csrRowPtrC_h[i]<<std::endl;

    int old_val, new_val;
    old_val = csrRowPtrC_h[0];
    csrRowPtrC_h[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrC_h[i];
        csrRowPtrC_h[i] = old_val + csrRowPtrC_h[i-1];
        old_val = new_val;
    }

    int nnzC = csrRowPtrC_h[m];

    control->accl_view.copy(csrRowPtrC_h, csrRowPtrC, (m + 1) * sizeof(int));
    
    int *csrColIndC = static_cast<int*>(matC->colIndices);
    T *csrValC = static_cast<T*>(matC->values);

    status2 = copy_Ct_to_C_general<T> (counter_one, csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt_d, csrColIndCt, queue_one_d, control);
    
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
