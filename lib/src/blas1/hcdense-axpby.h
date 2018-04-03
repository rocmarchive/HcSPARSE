#include "hcsparse.h"
#include "elementwise-operators.h"
#define BLOCK_SIZE 256

template <typename  T, ElementWiseOperator OP>
void axpby_kernel (const long size,
                   T *pR,
                   const long pROffset,
                   T *pX,
                   const long pXOffset,
                   const T *pY,
                   const long pYOffset,
                   const T *pAlpha,
                   const long pAlphaOffset,
                   const T *pBeta,
                   const long pBetaOffset,
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
            T beta = pBeta[pBetaOffset];
            pR[i + pROffset] = operation<T, OP>(pX[i + pXOffset] * alpha, pY[i + pYOffset] * beta);
        }
    }).wait();
}

template <typename T, ElementWiseOperator OP = EW_PLUS>
hcsparseStatus
axpby (hcdenseVector *r,
       const hcsparseScalar *alpha,
       const hcdenseVector *x,
       const hcsparseScalar *beta,
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
    T *avBeta = static_cast<T*>(beta->value);

    axpby_kernel<T, OP> (size, avR, r->offValues, avX, x->offValues, avY, y->offValues, avAlpha, alpha->offValue, avBeta, beta->offValue, globalSize, control);

    return hcsparseSuccess;
}
