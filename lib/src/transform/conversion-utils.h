#include "hcsparse.h"

template <typename T>
hcsparseStatus
indices_to_offsets (const int size,
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

    reduce_by_key<T> (size, &av_keys_output, av_csrOffsets, av_cooIndices, &av_values, control);

    exclusive_scan<T, EW_PLUS> (size, av_csrOffsets, av_csrOffsets, control);
}

