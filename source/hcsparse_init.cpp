#include "hcsparse.h"
#include "csrmv.hpp"
#include "scale.hpp"
#include "mm_reader.hpp"
#include "csr_meta.hpp"

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

// This function reads the file header at the given filepath, and returns the size
// of the sparse matrix in the hcsparseCooMatrix parameter.
// Post-condition: clears hcsparseCooMatrix, then sets pCooMatx->m, pCooMatx->n
// pCooMatx->nnz
hcsparseStatus
hcsparseHeaderfromFile( int* nnz, int* row, int* col, const char* filePath )
{

    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return hcsparseInvalid;
    }
    else
        return hcsparseInvalid;

   MatrixMarketReader< float > mm_reader;

    if( mm_reader.MMReadHeader( filePath ) )
        return hcsparseInvalid;

    *row = mm_reader.GetNumRows( );
    *col = mm_reader.GetNumCols( );
    *nnz = mm_reader.GetNumNonZeroes( );

    return hcsparseSuccess;
}

// This function reads the file at the given filepath, and returns the sparse
// matrix in the COO struct.  All matrix data is written to device memory
// Pre-condition: This function assumes that the device memory buffers have been
// pre-allocated by the caller
hcsparseStatus
hcsparseSCooMatrixfromFile( hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes )
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return hcsparseInvalid;
    }
    else
        return hcsparseInvalid;

    MatrixMarketReader< float > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return hcsparseInvalid;

    cooMatx->num_rows = mm_reader.GetNumRows( );
    cooMatx->num_cols = mm_reader.GetNumCols( );
    cooMatx->num_nonzeros = mm_reader.GetNumNonZeroes( );

    int *x = mm_reader.GetXCoordinates( );
    int *y = mm_reader.GetYCoordinates( );
    float *val = mm_reader.GetValCoordinates( );

    //JPA:: Coo matrix is need to be sorted as well because we need to have matrix
    // which is sorted by row and then column, in the mtx files usually is opposite.
    for(int i = 0 ;i < cooMatx->num_nonzeros; i++)
    {
        if(x[i] > x[i+1])
        {
            int tmp = x[i];
            x[i] = x[i+1];
            x[i+1] = tmp;
            int tmp1 = y[i];
            y[i] = y[i+1];
            y[i+1] = tmp1;
            float tmp2 = val[i];
            val[i] = val[i+1];
            val[i+1] = tmp2;
        }
        else if ( x[i] == x[i + 1])
        {
            if ( y[i] > y[i+1])
            {
                int tmp = x[i];
                x[i] = x[i+1];
                x[i+1] = tmp;
                int tmp1 = y[i];
                y[i] = y[i+1];
                y[i+1] = tmp1;
                float tmp2 = val[i];
                val[i] = val[i+1];
                val[i+1] = tmp2;
            }
        }
    }

    Concurrency::array_view<float> *values = static_cast<Concurrency::array_view<float> *>(cooMatx->values);
    Concurrency::array_view<int> *rowIndices = static_cast<Concurrency::array_view<int> *>(cooMatx->rowIndices);
    Concurrency::array_view<int> *colIndices = static_cast<Concurrency::array_view<int> *>(cooMatx->colIndices);

    for( int c = 0; c < cooMatx->num_nonzeros; ++c )
    {
        (*(rowIndices))[ c ] = x[ c ];
        (*(colIndices))[ c ] = y[ c ];
        (*(values))[ c ] = val[ c ];
    }

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseDCooMatrixfromFile( hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes )
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return hcsparseInvalid;
    }
    else
        return hcsparseInvalid;

    MatrixMarketReader< double > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return hcsparseInvalid;

    cooMatx->num_rows = mm_reader.GetNumRows( );
    cooMatx->num_cols = mm_reader.GetNumCols( );
    cooMatx->num_nonzeros = mm_reader.GetNumNonZeroes( );

    int *x = mm_reader.GetXCoordinates( );
    int *y = mm_reader.GetYCoordinates( );
    double *val = mm_reader.GetValCoordinates( );

    //JPA:: Coo matrix is need to be sorted as well because we need to have matrix
    // which is sorted by row and then column, in the mtx files usually is opposite.
    for(int i = 0 ;i < cooMatx->num_nonzeros; i++)
    {
        if(x[i] > x[i+1])
        {
            int tmp = x[i];
            x[i] = x[i+1];
            x[i+1] = tmp;
            int tmp1 = y[i];
            y[i] = y[i+1];
            y[i+1] = tmp1;
            double tmp2 = val[i];
            val[i] = val[i+1];
            val[i+1] = tmp2;
        }
        else if ( x[i] == x[i + 1])
        {
            if ( y[i] > y[i+1])
            {
                int tmp = x[i];
                x[i] = x[i+1];
                x[i+1] = tmp;
                int tmp1 = y[i];
                y[i] = y[i+1];
                y[i+1] = tmp1;
                double tmp2 = val[i];
                val[i] = val[i+1];
                val[i+1] = tmp2;
            }
        }
    }

    Concurrency::array_view<double> *values = static_cast<Concurrency::array_view<double> *>(cooMatx->values);
    Concurrency::array_view<int> *rowIndices = static_cast<Concurrency::array_view<int> *>(cooMatx->rowIndices);
    Concurrency::array_view<int> *colIndices = static_cast<Concurrency::array_view<int> *>(cooMatx->colIndices);

    for( int c = 0; c < cooMatx->num_nonzeros; ++c )
    {
        (*(rowIndices))[ c ] = x[ c ];
        (*(colIndices))[ c ] = y[ c ];
        (*(values))[ c ] = val[ c ];
    }

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseSCsrMatrixfromFile(hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes )
{

    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return hcsparseInvalid;
    }
    else
        return hcsparseInvalid;

    // Read data from a file on disk into CPU buffers
    // Data is read natively as COO format with the reader
    MatrixMarketReader< float > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return hcsparseInvalid;

    // JPA: Shouldn't that just be an assertion check? It seems to me that
    // the user have to call hcsparseHeaderfromFile before calling this function,
    // otherwise the whole pCsrMatrix will be broken;

    csrMatx->num_rows = mm_reader.GetNumRows( );
    csrMatx->num_cols = mm_reader.GetNumCols( );
    csrMatx->num_nonzeros = mm_reader.GetNumNonZeroes( );

    //  The following section of code converts the sparse format from COO to CSR
    int *x = mm_reader.GetXCoordinates( );
    int *y = mm_reader.GetYCoordinates( );
    float *val = mm_reader.GetValCoordinates( );

    for(int i = 0 ;i < csrMatx->num_nonzeros; i++)
    {
        if(x[i] > x[i+1])
        {
            int tmp = x[i];
            x[i] = x[i+1];
            x[i+1] = tmp;
            int tmp1 = y[i];
            y[i] = y[i+1];
            y[i+1] = tmp1;
            float tmp2 = val[i];
            val[i] = val[i+1];
            val[i+1] = tmp2;
        }
        else if ( x[i] == x[i + 1])
        {
            if ( y[i] > y[i+1])
            {
                int tmp = x[i];
                x[i] = x[i+1];
                x[i+1] = tmp;
                int tmp1 = y[i];
                y[i] = y[i+1];
                y[i+1] = tmp1;
                float tmp2 = val[i];
                val[i] = val[i+1];
                val[i+1] = tmp2;
            }
        }
    }

    Concurrency::array_view<float> *values = static_cast<Concurrency::array_view<float> *>(csrMatx->values);
    Concurrency::array_view<int> *rowOffsets = static_cast<Concurrency::array_view<int> *>(csrMatx->rowOffsets);
    Concurrency::array_view<int> *colIndices = static_cast<Concurrency::array_view<int> *>(csrMatx->colIndices);

    int current_row = 1;
    (*(rowOffsets))[ 0 ] = 0;
    for( int i = 0; i < csrMatx->num_nonzeros; i++ )
    {
        (*(colIndices))[ i ] = y[ i ];
        (*(values))[ i ] = val[ i ];

        while( x[ i ] >= current_row )
            (*(rowOffsets))[ current_row++ ] = i;
    }
    (*(rowOffsets))[ current_row ] = csrMatx->num_nonzeros;
    while( current_row <= csrMatx->num_rows )
        (*(rowOffsets))[ current_row++ ] = csrMatx->num_nonzeros;

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseDCsrMatrixfromFile( hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes )
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return hcsparseInvalid;
    }
    else
        return hcsparseInvalid;

    // Read data from a file on disk into CPU buffers
    // Data is read natively as COO format with the reader
    MatrixMarketReader< double > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return hcsparseInvalid;

    csrMatx->num_rows = mm_reader.GetNumRows( );
    csrMatx->num_cols = mm_reader.GetNumCols( );
    csrMatx->num_nonzeros = mm_reader.GetNumNonZeroes( );

    //  The following section of code converts the sparse format from COO to CSR
    int *x = mm_reader.GetXCoordinates( );
    int *y = mm_reader.GetYCoordinates( );
    double *val = mm_reader.GetValCoordinates( );

    for(int i = 0 ;i < csrMatx->num_nonzeros; i++)
    {
        if(x[i] > x[i+1])
        {
            int tmp = x[i];
            x[i] = x[i+1];
            x[i+1] = tmp;
            int tmp1 = y[i];
            y[i] = y[i+1];
            y[i+1] = tmp1;
            double tmp2 = val[i];
            val[i] = val[i+1];
            val[i+1] = tmp2;
        }
        else if ( x[i] == x[i + 1])
        {
            if ( y[i] > y[i+1])
            {
                int tmp = x[i];
                x[i] = x[i+1];
                x[i+1] = tmp;
                int tmp1 = y[i];
                y[i] = y[i+1];
                y[i+1] = tmp1;
                double tmp2 = val[i];
                val[i] = val[i+1];
                val[i+1] = tmp2;
            }
        }
    }

    Concurrency::array_view<double> *values = static_cast<Concurrency::array_view<double> *>(csrMatx->values);
    Concurrency::array_view<int> *rowOffsets = static_cast<Concurrency::array_view<int> *>(csrMatx->rowOffsets);
    Concurrency::array_view<int> *colIndices = static_cast<Concurrency::array_view<int> *>(csrMatx->colIndices);

    int current_row = 1;
    (*(rowOffsets))[ 0 ] = 0;
    for( int i = 0; i < csrMatx->num_nonzeros; i++ )
    {
        (*(colIndices))[ i ] = y[ i ];
        (*(values))[ i ] = val[ i ];

        while( x[ i ] >= current_row )
            (*(rowOffsets))[ current_row++ ] = i;
    }
    (*(rowOffsets))[ current_row ] = csrMatx->num_nonzeros;
    while( current_row <= csrMatx->num_rows )
        (*(rowOffsets))[ current_row++ ] = csrMatx->num_nonzeros;

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseCsrMetaSize( hcsparseCsrMatrix* csrMatx, hcsparseControl *control )
{
    Concurrency::array_view<int> *rCsrRowOffsets = static_cast<Concurrency::array_view<int> *>(csrMatx->rowOffsets);
    int * dataRO = rCsrRowOffsets->data();

    csrMatx->rowBlockSize = ComputeRowBlocksSize( dataRO, csrMatx->num_rows, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR );

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseCsrMetaCompute( hcsparseCsrMatrix* csrMatx, hcsparseControl *control )
{
    // Check to ensure nRows can fit in 32 bits
    if( static_cast<ulong>( csrMatx->num_rows ) > static_cast<ulong>( std::pow( 2, ( 64 - ROWBITS ) ) ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present ((64-WG_BITS) bits) !" );
        return hcsparseInvalid;
    }

    Concurrency::array_view<int> *rCsrRowOffsets = static_cast<Concurrency::array_view<int> *>(csrMatx->rowOffsets);
    int *dataRO = rCsrRowOffsets->data();
    Concurrency::array_view<ulong> *rRowBlocks = static_cast<Concurrency::array_view<ulong> *>(csrMatx->rowBlocks);
    ulong *dataRB = rRowBlocks->data();

    ComputeRowBlocks( dataRB, csrMatx->rowBlockSize, dataRO, csrMatx->num_rows, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, true );

    return hcsparseSuccess;
}
