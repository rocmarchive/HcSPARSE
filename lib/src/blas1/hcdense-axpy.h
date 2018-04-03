#include "hcsparse.h"
#include "elementwise-operators.h"
#define BLOCK_SIZE 256

template <typename  T, ElementWiseOperator OP>
void axpy_kernel (const long size,
                  T *pR,
                  const long pROffset,
                  T *pX,
                  const long pXOffset,
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
            T alpha = pAlpha[pAlphaOffset];
            pR[i + pROffset] = operation<T, OP>(pX[i + pXOffset] * alpha, pY[i + pYOffset]);
        }
    }).wait();
}

template <typename T, ElementWiseOperator OP = EW_PLUS>
hcsparseStatus
axpy (hcdenseVector *r,
      const hcsparseScalar *alpha,
      const hcdenseVector *x,
      const hcdenseVector* y,
      hcsparseControl* control)
{
    int size = r->num_values;
    int blocksNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalSize = blocksNum * BLOCK_SIZE;

    T *avR = static_cast<T*>(r->values);
    T *avX = static_cast<T*>(x->values);
    T *avY = static_cast<T*>(y->values);
    T *avAlpha = static_cast<T*>(alpha->value);

    axpy_kernel<T, OP> (size, avR, r->offValues, avX, x->offValues, avY, y->offValues, avAlpha, alpha->offValue, globalSize, control);

    return hcsparseSuccess;
}
