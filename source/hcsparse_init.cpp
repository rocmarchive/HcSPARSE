#include "hcsparse.h"
#include "csrmv.hpp"
#include "scale.hpp"

int hcsparseInitialized = 0;

hcsparseStatus
hcsparseSetup(void)
{
    if(hcsparseInitialized)
    {
        return hcsparseSuccess;
    }

    hcsparseInitialized = 1;
    return hcsparseSuccess;
}

hcsparseStatus
hcsparseTeardown(void)
{
    if(!hcsparseInitialized)
    {
        return hcsparseSuccess;
    }

    hcsparseInitialized = 0;
    return hcsparseSuccess;
}

// Convenience sparse matrix construction functions
hcsparseStatus
hcsparseInitScalar( hcsparseScalar* scalar )
{
    scalar->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitVector( hcdenseVector* vec )
{
    vec->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitCooMatrix( hcsparseCooMatrix* cooMatx )
{
    cooMatx->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitCsrMatrix( hcsparseCsrMatrix* csrMatx )
{
    csrMatx->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcdenseInitMatrix( hcdenseMatrix* denseMatx )
{
    denseMatx->clear();

    return hcsparseSuccess;
}

hcsparseStatus
    hcsparseScsrmv( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        const hcsparseControl* control )
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    //check opencl elements
    if (x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csrmv<float>(alpha, matx, x, beta, y, control);
}

hcsparseStatus
    hcsparseDcsrmv( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        const hcsparseControl* control )
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    //check opencl elements
    if (x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csrmv<double>(alpha, matx, x, beta, y, control);
}

hcsparseStatus
hcdenseSscale ( hcdenseVector* r,
                const hcsparseScalar* alpha,
                const hcdenseVector* y,
                const hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return scale<float> (r, alpha, y, control);
}


hcsparseStatus
hcdenseDscale( hcdenseVector* r,
               const hcsparseScalar* alpha,
               const hcdenseVector* y,
               const hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return scale<double> (r, alpha, y, control);
}

