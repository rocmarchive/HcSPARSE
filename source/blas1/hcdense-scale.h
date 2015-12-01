#ifndef _HCSPARSE_SCALE_H_
#define _HCSPARSE_SCALE_H_

#include "hcsparse.h"

template <typename T>
hcsparseStatus
scale ( hcdenseVector* r,
                const hcsparseScalar* alpha,
                const hcdenseVector* y,
                const hcsparseControl* control);
#endif
