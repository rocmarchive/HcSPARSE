#include "hcsparse.h"

#define WAVE_SIZE 64
#define GROUP_SIZE 256

template <typename T>
hcsparseStatus
indices_to_offsets (const int num_rows,
                    const int size,
                    Concurrency::array_view<T> &av_csrOffsets,
                    const Concurrency::array_view<T> &av_cooIndices,
                    const hcsparseControl* control)
{
    T *keys_output = (T*) calloc (size, sizeof(T));
    T *values = (T*) calloc (size, sizeof(T));

    for (int i = 0; i < size; i++)
        values[i] = 1;

    Concurrency::array_view<T> av_values(size, values);
    Concurrency::array_view<T> av_keys_output(size, keys_output);

    reduce_by_key<T> (size, av_keys_output, av_csrOffsets, av_cooIndices, av_values, control);

    exclusive_scan<T, EW_PLUS> (num_rows, av_csrOffsets, av_csrOffsets, control);

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
offsets_to_indices(const int num_rows,
                   const int size,
                   Concurrency::array_view<T> &av_cooIndices,
                   const Concurrency::array_view<T> &av_csrOffsets,
                   const hcsparseControl* control)
{
    int subwave_size = WAVE_SIZE;

    int elements_per_row = size / num_rows; // assumed number elements per row;

    // adjust subwave_size according to elements_per_row;
    // each wavefront will be assigned to process to the row of the csr matrix
    if(WAVE_SIZE > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (elements_per_row < 64) {  subwave_size = 32;  }
    }
    if (elements_per_row < 32) {  subwave_size = 16;  }
    if (elements_per_row < 16) {  subwave_size = 8;  }
    if (elements_per_row < 8)  {  subwave_size = 4;  }
    if (elements_per_row < 4)  {  subwave_size = 2;  }

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    int predicted = subwave_size * num_rows;

    int global_work_size = GROUP_SIZE * ((predicted + GROUP_SIZE - 1 ) / GROUP_SIZE);

    Concurrency::extent<1> grdExt(global_work_size);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);

    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        const int global_id   = tidx.global[0];         
        const int local_id    = tidx.local[0];          
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size; 
        const int num_vectors = t_ext[0] / subwave_size;
        for(int row = vector_id; row < num_rows; row += num_vectors)
        {
            const int row_start = av_csrOffsets[row];
            const int row_end   = av_csrOffsets[row+1];
            for(int j = row_start + thread_lane; j < row_end; j += subwave_size)
                av_cooIndices[j] = row;
        }
    });

    return hcsparseSuccess;
}
