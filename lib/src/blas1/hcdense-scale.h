#include "hcsparse.h"
#define BLOCK_SIZE 256

template <typename  T>
void scale_kernel (const long size,
              hc::array_view<T> &pR,
              const long pROffset,
              const hc::array_view<T> &pY,
              const long pYOffset,
              const hc::array_view<T> &pAlpha,
              const long pAlphaOffset,
              const int globalSize,
              const hcsparseControl* control)
{
    hc::extent<1> grdExt( globalSize );
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu))
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

    hc::array_view<T> *avR = static_cast<hc::array_view<T>*>(r->values);
    hc::array_view<T> *avY = static_cast<hc::array_view<T>*>(y->values);
    hc::array_view<T> *avAlpha = static_cast<hc::array_view<T>*>(alpha->value);

    scale_kernel<T> (size, *avR, r->offValues, *avY, y->offValues, *avAlpha, alpha->offValue, globalSize, control);

    return hcsparseSuccess;
}
