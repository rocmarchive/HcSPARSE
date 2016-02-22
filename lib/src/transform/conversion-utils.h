#include "hcsparse.h"

#define WAVE_SIZE 64
#define GROUP_SIZE 256

template <typename T>
hcsparseStatus
indices_to_offsets (const int num_rows,
                    const int size,
                    hc::array_view<T> &av_csrOffsets,
                    const hc::array_view<T> &av_cooIndices,
                    const hcsparseControl* control)
{
    T *keys_output = (T*) calloc (size, sizeof(T));
    T *values = (T*) calloc (size, sizeof(T));

    for (int i = 0; i < size; i++)
        values[i] = 1;

    hc::array_view<T> av_values(size, values);
    hc::array_view<T> av_keys_output(size, keys_output);

    reduce_by_key<T> (size, av_keys_output, av_csrOffsets, av_cooIndices, av_values, control);

    exclusive_scan<T, EW_PLUS> (num_rows, av_csrOffsets, av_csrOffsets, control);

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
offsets_to_indices (const int num_rows,
                    const int size,
                    hc::array_view<T> &av_cooIndices,
                    const hc::array_view<T> &av_csrOffsets,
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

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        const int global_id   = tidx.global[0];
        const int local_id    = tidx.local[0];
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size;
        const int num_vectors = grdExt[0] / subwave_size;
        for(int row = vector_id; row < num_rows; row += num_vectors)
        {
            const int row_start = av_csrOffsets[row];
            const int row_end   = av_csrOffsets[row+1];
            for(int j = row_start + thread_lane; j < row_end; j += subwave_size)
                av_cooIndices[j] = row;
        }
    }).wait();

    return hcsparseSuccess;
}

template<typename T>
hcsparseStatus
transform_csr_2_dense (int size,
                       const hc::array_view<int> &row_offsets,
                       const hc::array_view<int> &col_indices,
                       const hc::array_view<T> &values,
                       const int num_rows,
                       const int num_cols,
                       hc::array_view<T> &A,
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

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        const int global_id   = tidx.global[0];
        const int local_id    = tidx.local[0];
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size;
        const int num_vectors = grdExt[0] / subwave_size;
        for(int row = vector_id; row < num_rows; row += num_vectors)
        {
            const int row_start = row_offsets[row];
            const int row_end   = row_offsets[row+1];
            for(int j = row_start + thread_lane; j < row_end; j += subwave_size)
                A[row * num_cols + col_indices[j]] = values[j];
        }
    }).wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
calculate_num_nonzeros (int dense_size,
                        const hc::array_view<T> &A,
                        hc::array_view<int> &nnz_locations,
                        int& num_nonzeros,
                        const hcsparseControl* control)
{
    int global_work_size = 0;

    if (dense_size % GROUP_SIZE == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / GROUP_SIZE * GROUP_SIZE + GROUP_SIZE;

    if (dense_size < GROUP_SIZE) global_work_size = GROUP_SIZE;

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        int index = tidx.global[0];
        if (index < dense_size)
        {
            if (A[index] != 0)
                nnz_locations[index] = 1;
            else
                nnz_locations[index] = 0;
        }
    }).wait();

    int* nnz_buf = (int*) calloc (1, sizeof(int));
    hc::array_view<int> av_nnz(1, nnz_buf);

    hcsparseScalar nnz;
    nnz.value = &av_nnz;
    nnz.offValue = 0;

    hcdenseVector nnz_location_vec;
    nnz_location_vec.num_values = dense_size;
    nnz_location_vec.values = &nnz_locations;
    nnz_location_vec.offValues = 0;

    reduce<int, RO_PLUS>(&nnz, &nnz_location_vec, control);

    num_nonzeros = av_nnz[0];

    free(nnz_buf);
    return hcsparseSuccess;
}

template<typename T>
hcsparseStatus
dense_to_coo (int dense_size,
              int num_cols,
              hc::array_view<int> &row_indices,
              hc::array_view<int> &col_indices,
              hc::array_view<T> &values,
              const hc::array_view<T> &A,
              const hc::array_view<int> &nnz_locations,
              const hc::array_view<int> &coo_indexes,
              const hcsparseControl* control)
{
    int global_work_size = 0;

    if (dense_size % GROUP_SIZE == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / GROUP_SIZE * GROUP_SIZE + GROUP_SIZE;

    if (dense_size < GROUP_SIZE) global_work_size = GROUP_SIZE;

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) __attribute__((hc, cpu))
    {
        int index = tidx.global[0];
        if (nnz_locations[index] == 1 && index < dense_size)
        {
            int row_index = index / num_cols;
            int col_index = index % num_cols;
            int location = coo_indexes[index];
            row_indices[ location ] = row_index;
            col_indices[ location ] = col_index;
            values [ location ] = A[index];
        }
    }).wait();

    return hcsparseSuccess;
}

