#include "hcsparse.h"
#define BLOCK_SIZE 256

template <typename  T>
void scale_kernel (const long size,
                   T *pR,
                   const long pROffset,
                   const T *pY,
                   const long pYOffset,
                   const T *pAlpha,
                   const long pAlphaOffset,
                   const int globalSize,
                   hcsparseControl* control)
{
    hc::extent<1> grdExt( globalSize );
    hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) [[hc]]
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
scale (hcdenseVector* r,
       const hcsparseScalar* alpha,
       const hcdenseVector* y,
       hcsparseControl* control)
{
    int size = r->num_values;
    int blocksNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalSize = blocksNum * BLOCK_SIZE;

    T *avR = static_cast<T*>(r->values);
    T *avY = static_cast<T*>(y->values);
    T *avAlpha = static_cast<T*>(alpha->value);

    scale_kernel<T> (size, avR, r->offValues, avY, y->offValues, avAlpha, alpha->offValue, globalSize, control);

    return hcsparseSuccess;
}
