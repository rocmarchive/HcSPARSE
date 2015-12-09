#ifndef _HCSPARSE_AXPBY_H_
#define _HCSPARSE_AXPBY_H_

#include "hcsparse.h"

template <typename T>
hcsparseStatus
axpby (hcdenseVector *r,
               const hcsparseScalar *alpha,
               const hcdenseVector *x,
               const hcsparseScalar *beta,
               const hcdenseVector* y,
               const hcsparseControl* control);
#endif
