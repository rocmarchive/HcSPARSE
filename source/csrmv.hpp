#include "hcsparse.h"

template<typename T>
hcsparseStatus
csrmv_vector(const hcsparseScalar* pAlpha,
             const hcsparseCsrMatrix* pMatx,
             const hcdenseVector* pX,
             const hcsparseScalar* pBeta,
             hcdenseVector* pY,
             hcsparseControl control)
{
    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv_adaptive( const hcsparseScalar* pAlpha,
                const hcsparseCsrMatrix* pCsrMatx,
                const hcdenseVector* pX,
                const hcsparseScalar* pBeta,
                hcdenseVector* pY,
                hcsparseControl control )
{
    return hcsparseSuccess;
}

template <typename T>
hcsparseStatus
csrmv (const hcsparseScalar *pAlpha,
       const hcsparseCsrMatrix *pCsrMatx,
       const hcdenseVector *pX,
       const hcsparseScalar *pBeta,
       hcdenseVector *pY,
       hcsparseControl control)
{
    if( (pCsrMatx->rowBlocks == nullptr) && (pCsrMatx->rowBlockSize == 0) )
    {
        // Call Vector CSR Kernels
        return csrmv_vector<T>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
    }
    else
    {
        if( ( pCsrMatx->rowBlocks == nullptr ) || ( pCsrMatx->rowBlockSize == 0 ) )
        {
            // rowBlockSize varible is not zero but no pointer
            return hcsparseInvalid;
        }

        // Call adaptive CSR kernels
        return csrmv_adaptive<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );

    }
}
