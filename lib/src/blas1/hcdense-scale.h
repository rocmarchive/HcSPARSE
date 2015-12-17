#include "../hcsparse.h"
#define BLOCK_SIZE 256

template <typename  T>
void scale_kernel (const long size,
              Concurrency::array_view<T> &pR,
              const long pROffset,
              const Concurrency::array_view<T> &pY,
              const long pYOffset,
              const Concurrency::array_view<T> &pAlpha,
              const long pAlphaOffset,
              const int globalSize,
              const hcsparseControl* control)
{
    Concurrency::extent<1> grdExt( globalSize );
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);
    Concurrency::parallel_for_each(control->accl_view, t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
        int i = tidx.global[0];
        if (i < size)
        {
            long alpha = pAlpha[pAlphaOffset];
            pR[i + pROffset] = pY[i + pYOffset]* alpha;
        }
    });
}

template <typename T>
hcsparseStatus
scale ( hcdenseVector* r,
                const hcsparseScalar* alpha,
                const hcdenseVector* y,
                const hcsparseControl* control)
{
    int size = r->num_values;
    int blocksNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalSize = blocksNum * BLOCK_SIZE;

    Concurrency::array_view<T> *avR = static_cast<Concurrency::array_view<T>*>(r->values);
    Concurrency::array_view<T> *avY = static_cast<Concurrency::array_view<T>*>(y->values);
    Concurrency::array_view<T> *avAlpha = static_cast<Concurrency::array_view<T>*>(alpha->value);

    scale_kernel<T> (size, *avR, r->offValues, *avY, y->offValues, *avAlpha, alpha->offValue, globalSize, control);

    return hcsparseSuccess;
}
