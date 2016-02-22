#include "hcsparse.h"
template <typename T>
hcsparseStatus
Norm2 (hcsparseScalar* pS,
       const hcdenseVector* pX,
       const hcsparseControl* control)
{
    return reduce<T, RO_SQR, RO_SQRT>(pS, pX, control);
}
