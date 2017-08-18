#ifndef MMIO_WRAPPER_H
#define MMIO_WRAPPER_H

#include "../src/io/mm_reader.h"
#include <cstdlib>
#include <algorithm>

// 1. hcsparseCsrMatrixfromFile

// This function reads header and values from the given input matrix file and 
// fills the values, rowOffsets, colIndices array

// Return Values
// --------------------------------------------------------------------------
// -1            Invalid parameters provided/Internal error
//  0            Return Successfully

template <typename T>
int hcsparseCsrMatrixfromFile (const char* filePath,
                                bool read_explicit_zeroes,
                                T **values,
                                int **rowOffsets,
                                int **colIndices,
                                int *num_rows,
                                int *num_cols,
                                int *num_nonzeros)
{

    std::cout << "file : " << filePath <<std::endl;
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return -1;
    }
    else
        return -1;

    std::string filename = filePath;
    FILE *mm_file = fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        perror( "Cannot Open Matrix-Market File !\n" );
        return -1;
    }

    MatrixMarketReader<T> mm_reader;
    if( mm_reader.MMReadHeader( filePath ) )
        return -1;

    *num_rows = mm_reader.GetNumRows( );
    *num_cols = mm_reader.GetNumCols( );
    *num_nonzeros = mm_reader.GetNumNonZeroes( );

    std::cout<<"nRow" << *num_rows << "nCols/: " << *num_cols << "nnz : " << *num_nonzeros << std::endl; 

    T *tempValues;
    int *tempRowOffsets, *tempColIndices;

    tempValues = (T*)calloc(*num_nonzeros, sizeof(T));
    tempRowOffsets = (int*)calloc((*num_rows)+1, sizeof(int));
    tempColIndices = (int*)calloc(*num_nonzeros, sizeof(int));

    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) ) {
        std::cout << "Error in ReadFormat " << std::endl;
        return -1;
    }

    // JPA: Shouldn't that just be an assertion check? It seems to me that
    // the user have to call hcsparseHeaderfromFile before calling this function,
    // otherwise the whole pCsrMatrix will be broken;

    int actual_rows = mm_reader.GetNumRows( );
    int actual_cols = mm_reader.GetNumCols( );
    int actual_num_nonzeros = mm_reader.GetNumNonZeroes( );

    std::cout<<"nRow" << actual_rows << "nCols/: " << actual_cols << "nnz: " << actual_num_nonzeros << std::endl; 

    Coordinate<T>* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + actual_num_nonzeros, CoordinateCompare<T> );

    int current_row = 0;
    tempRowOffsets[ 0 ] = 0;
    for( int i = 0; i < actual_num_nonzeros; i++ )
    {
        tempColIndices[ i ] = coords[i].y;
        tempValues[ i ] = coords[i].val;

        while( coords[i].x >= current_row )
            tempRowOffsets[ current_row++ ] = i;
    }
    tempRowOffsets[ current_row ] = (actual_num_nonzeros);
    while( current_row <= actual_rows )
        tempRowOffsets[ current_row++ ] = (actual_num_nonzeros);

    *values = (T*)calloc(*num_nonzeros, sizeof(T));
    *rowOffsets = (int*)calloc((*num_rows)+1, sizeof(int));
    *colIndices = (int*)calloc(*num_nonzeros, sizeof(int));
    memcpy(*values, tempValues, sizeof(T) * (*num_nonzeros));
    memcpy(*rowOffsets, tempRowOffsets, sizeof(int) * (*num_rows+1));
    memcpy(*colIndices, tempColIndices, sizeof(int) * (*num_nonzeros));

    std::cout << "dest_values : 5 " << (*values)[5] << " temp : " << tempValues[5] << std::endl;
    free(tempValues);
    free(tempRowOffsets);
    free(tempColIndices);

    return 0;

}



#endif
