#include "hcsparse.h"

#define WAVE_SIZE 64
#define SUBWAVE_SIZE 64
#define WG_SIZE 256
#define INDEX_TYPE int
#define SIZE_TYPE ulong
#define VALUE_TYPE float
#define GLOBAL_SIZE WG_SIZE
#define EXTENDED_PRECISION 1
#define ROWS_FOR_VECTOR 1
#define BLOCK_MULTIPLIER 3
#define BLOCKSIZE 1024
#define WGBITS 24
#define ROWBITS 32

#ifndef INDEX_TYPE
#error "INDEX_TYPE undefined!"
#endif

#ifndef VALUE_TYPE
#error "VALUE_TYPE undefined!"
#endif

#ifndef SIZE_TYPE
#error "SIZE_TYPE undefined!"
#endif

#ifndef WG_SIZE
#error "WG_SIZE undefined!"
#endif

#ifndef WAVE_SIZE
#error "WAVE_SIZE undefined!"
#endif

#ifndef SUBWAVE_SIZE
#error "SUBWAVE_SIZE undefined!"
#endif

#if ( (SUBWAVE_SIZE > WAVE_SIZE) || (SUBWAVE_SIZE != 2 && SUBWAVE_SIZE != 4 && SUBWAVE_SIZE != 8 && SUBWAVE_SIZE != 16 && SUBWAVE_SIZE != 32 && SUBWAVE_SIZE != 64) )
#error "SUBWAVE_SIZE is not  a power of two!"
#endif

template<typename T>
hcsparseStatus
csrmv_vector(const hcsparseScalar* pAlpha,
             const hcsparseCsrMatrix* pMatx,
             const hcdenseVector* pX,
             const hcsparseScalar* pBeta,
             hcdenseVector* pY,
             hcsparseControl control)
{
    uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
#if WAVE_SIZE > 32
{
    //this apply only for devices with wavefront > 32 like AMD(64)
    if (nnz_per_row < 64) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 32
}
}
#endif
    if (nnz_per_row < 32) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 16
}
    if (nnz_per_row < 16) {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 8
}
    if (nnz_per_row < 8)  {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 4
}
    if (nnz_per_row < 4)  {
#undef SUBWAVE_SIZE
#define SUBWAVE_SIZE 2
}

    if(typeid(T) == typeid(double))
    {
#undef VALUE_TYPE
#define VALUE_TYPE double
    }
    else if(typeid(T) == typeid(float))
    {
#undef VALUE_TYPE
#define VALUE_TYPE float
    }

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    uint predicted = SUBWAVE_SIZE * pMatx->num_rows;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of WG_SIZE. Don't know if that
    // have an impact on performance
    uint global_work_size =
            WG_SIZE * ((predicted + WG_SIZE - 1 ) / WG_SIZE);

    if( global_work_size < WG_SIZE)
    {
#define GLOBAL_SIZE WG_SIZE
    }

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
