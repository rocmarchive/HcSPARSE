#include "hcsparse.h"

#define WAVE_SIZE 64
#define GROUP_SIZE 256

template <typename T>
hcsparseStatus
indices_to_offsets (const int num_rows,
                    const int size,
                    T *av_csrOffsets,
                    const T *av_cooIndices,
                    hcsparseControl* control)
{
    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *values = (T*) calloc (size, sizeof(T));

    for (int i = 0; i < size; i++)
        values[i] = 1;

    T *av_values = (T*) am_alloc(size * sizeof(T), acc, 0);
    T *av_keys_output = (T*) am_alloc(size * sizeof(T), acc, 0);

    control->accl_view.copy(values, av_values, size * sizeof(T));

    reduce_by_key<T> (size, av_keys_output, av_csrOffsets, av_cooIndices, av_values, control);
    control->accl_view.wait();

    exclusive_scan<T, EW_PLUS> (num_rows, av_csrOffsets, av_csrOffsets, control);
    control->accl_view.wait();

    free(values);
    am_free(av_values);
    am_free(av_keys_output);

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
offsets_to_indices (const int num_rows,
                    const int size,
                    T *av_cooIndices,
                    const T *av_csrOffsets,
                    hcsparseControl* control)
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

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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
transform_csr_2_dense (ulong size,
                       const int *row_offsets,
                       const int *col_indices,
                       const T *values,
                       const int num_rows,
                       const int num_cols,
                       T *A,
                       hcsparseControl* control)
{
    int subwave_size = WAVE_SIZE;

    ulong elements_per_row = size / num_rows; // assumed number elements per row;

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

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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
    control->accl_view.wait();

    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
calculate_num_nonzeros (ulong dense_size,
                        const T *A,
                        int *nnz_locations1,
                        int& num_nonzeros,
                        hcsparseControl* control)
{
    hc::accelerator acc = (control->accl_view).get_accelerator();

    int *nnz_locations = (int*) am_alloc(dense_size * sizeof(int), acc, 0);

    int global_work_size = 0;

    if (dense_size % GROUP_SIZE == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / GROUP_SIZE * GROUP_SIZE + GROUP_SIZE;

    if (dense_size < GROUP_SIZE) global_work_size = GROUP_SIZE;

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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

    control->accl_view.copy(nnz_locations, nnz_locations1, dense_size * sizeof(int));

    hcsparseScalar nnz;
    nnz.value  = (int*) am_alloc(1 * sizeof(int), acc, 0);
    nnz.offValue = 0;

    hcdenseVector nnz_location_vec;
    nnz_location_vec.num_values = dense_size;
    nnz_location_vec.values = (int*) am_alloc(dense_size * sizeof(int), acc, 0);
    nnz_location_vec.offValues = 0;

    control->accl_view.copy(nnz_locations, nnz_location_vec.values, dense_size * sizeof(int));

    reduce<int, RO_PLUS>(&nnz, &nnz_location_vec, control);

    control->accl_view.copy(nnz.value, &num_nonzeros, 1 * sizeof(int));

    am_free(nnz.value);
    am_free(nnz_location_vec.values);
    am_free(nnz_locations);

    return hcsparseSuccess;
}

template<typename T>
hcsparseStatus
dense_to_coo (ulong dense_size,
              int num_cols,
              int *row_indices,
              int *col_indices,
              T *values,
              const T *A,
              const int *nnz_locations,
              const int *coo_indexes,
              hcsparseControl* control)
{
    int global_work_size = 0;

    if (dense_size % GROUP_SIZE == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / GROUP_SIZE * GROUP_SIZE + GROUP_SIZE;

    if (dense_size < GROUP_SIZE) global_work_size = GROUP_SIZE;

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
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

template<typename T>
hcsparseStatus
transform_csc_2_dense (ulong size,
                       const int *col_offsets,
                       const int *row_indices,
                       const T *values,
                       const int num_rows,
                       const int num_cols,
                       T *A,
                       hcsparseControl* control)
{
    int subwave_size = WAVE_SIZE;

    ulong elements_per_col = size / num_cols; // assumed number elements per col;

    // adjust subwave_size according to elements_per_col;
    // each wavefront will be assigned to process to the col of the csr matrix
    if(WAVE_SIZE > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (elements_per_col < 64) {  subwave_size = 32;  }
    }
    if (elements_per_col < 32) {  subwave_size = 16;  }
    if (elements_per_col < 16) {  subwave_size = 8;  }
    if (elements_per_col < 8)  {  subwave_size = 4;  }
    if (elements_per_col < 4)  {  subwave_size = 2;  }

    // subwave takes care of each col in matrix;
    // predicted number of subwaves to be executed;
    int predicted = subwave_size * num_cols;

    int global_work_size = GROUP_SIZE * ((predicted + GROUP_SIZE - 1 ) / GROUP_SIZE);

    hc::extent<1> grdExt(global_work_size);
    hc::tiled_extent<1> t_ext = grdExt.tile(GROUP_SIZE);

    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        const int global_id   = tidx.global[0];
        const int local_id    = tidx.local[0];
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size;
        const int num_vectors = grdExt[0] / subwave_size;
        for(int col = vector_id; col < num_cols; col += num_vectors)
        {
            const int col_start = col_offsets[col];
            const int col_end   = col_offsets[col+1];
            for(int j = col_start + thread_lane; j < col_end; j += subwave_size)
                A[row_indices[j] * num_cols + col] = values[j];
        }
    }).wait();
    control->accl_view.wait();

    return hcsparseSuccess;
}

