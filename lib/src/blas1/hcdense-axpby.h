#include "hcsparse.h"
#include "elementwise-operators.h"
#define BLOCK_SIZE 256

template <typename  T, ElementWiseOperator OP>
void axpby_kernel (const long size,
              Concurrency::array_view<T> &pR,
              const long pROffset,
              Concurrency::array_view<T> &pX,
              const long pXOffset,
              const Concurrency::array_view<T> &pY,
              const long pYOffset,
              const Concurrency::array_view<T> &pAlpha,
              const long pAlphaOffset,
              const Concurrency::array_view<T> &pBeta,
              const long pBetaOffset,
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
            T alpha = pAlpha[pAlphaOffset];
            T beta = pBeta[pBetaOffset];
            pR[i + pROffset] = operation<T, OP>(pX[i + pXOffset] * alpha, pY[i + pYOffset] * beta);
        }
    });
}

template <typename T, ElementWiseOperator OP = EW_PLUS>
hcsparseStatus
axpby (hcdenseVector *r,
               const hcsparseScalar *alpha,
               const hcdenseVector *x,
               const hcsparseScalar *beta,
               const hcdenseVector* y,
               const hcsparseControl* control)
{
    int size = r->num_values;
    int blocksNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalSize = blocksNum * BLOCK_SIZE;

    Concurrency::array_view<T> *avR = static_cast<Concurrency::array_view<T>*>(r->values);
    Concurrency::array_view<T> *avX = static_cast<Concurrency::array_view<T>*>(x->values);
    Concurrency::array_view<T> *avY = static_cast<Concurrency::array_view<T>*>(y->values);
    Concurrency::array_view<T> *avAlpha = static_cast<Concurrency::array_view<T>*>(alpha->value);
    Concurrency::array_view<T> *avBeta = static_cast<Concurrency::array_view<T>*>(beta->value);

    axpby_kernel<T, OP> (size, *avR, r->offValues, *avX, x->offValues, *avY, y->offValues, *avAlpha, alpha->offValue, *avBeta, beta->offValue, globalSize, control);

    return hcsparseSuccess;
}
