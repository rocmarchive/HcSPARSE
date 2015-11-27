#ifndef MM_IO_H
#define MM_IO_H

#include "hcsparse.h"
/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((typecode)[0]='M')
#define mm_set_coordinate(typecode)	((typecode)[1]='C')
#define mm_set_array(typecode)	((typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)
#define mm_set_sparse(typecode)	mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((typecode)[2]='C')
#define mm_set_real(typecode)	((typecode)[2]='R')
#define mm_set_pattern(typecode)((typecode)[2]='P')
#define mm_set_integer(typecode)((typecode)[2]='I')


#define mm_set_symmetric(typecode)((typecode)[3]='S')
#define mm_set_general(typecode)((typecode)[3]='G')
#define mm_set_skew(typecode)	((typecode)[3]='K')
#define mm_set_hermitian(typecode)((typecode)[3]='H')

#define mm_clear_typecode(typecode) ((typecode)[0]=(typecode)[1]= \
				(typecode)[2]=' ',(typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/********************* Matrix Market error codes ***************************/

#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF	12
#define MM_NOT_MTX		13
#define MM_NO_HEADER		14
#define MM_UNSUPPORTED_TYPE	15
#define MM_LINE_TOO_LONG	16
#define MM_COULD_NOT_WRITE_FILE 17

#define MM_MTX_STR	"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSE_STR	"coordinate"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR	"real"
#define MM_INT_STR	"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR	"symmetric"
#define MM_HERM_STR	"hermitian"
#define MM_SKEW_STR	"skew-symmetric"
#define MM_PATTERN_STR  "pattern"

#define MM_MAX_LINE_LENGTH 1025
#define MM_MAX_TOKEN_LENGTH 64
#define MatrixMarketBanner "%%MatrixMarket"
#define MAX_RAND_VAL 5.0

template <typename FloatType>
class MatrixMarketReader
{
    char Typecode[ 4 ];
    int nNZ;
    int nRows;
    int nCols;
    int isSymmetric;
    int isDoubleMem;
    int *x;
    int *y;
    FloatType *val;

public:
    MatrixMarketReader( ): nNZ( 0 ), nRows( 0 ), nCols( 0 ), isSymmetric( 0 ), isDoubleMem( 0 )
    {
        for( auto c : Typecode )
            c = '\0';

        x = NULL;
        y = NULL;
        val = NULL;
    }

    int MMReadBanner( FILE* infile );
    int MMReadMtxCrdSize( FILE* infile );

    int GetNumRows( )
    {
        return nRows;
    }

    int GetNumCols( )
    {
        return nCols;
    }

    int GetNumNonZeroes( )
    {
        return nNZ;
    }

    int GetSymmetric( )
    {
        return isSymmetric;
    }

    char &GetTypecode( )
    {
        return Typecode;
    }

    int *GetXCoordinates( )
    {
        return x;
    }

    int *GetYCoordinates( )
    {
        return y;
    }

    FloatType *GetValCoordinates( )
    {
        return val;
    }

    ~MatrixMarketReader( )
    {
        delete[ ] x;
        delete[ ] y;
        delete[ ] val;
    }
};

template<typename FloatType>
int MatrixMarketReader<FloatType>::MMReadBanner( FILE *infile )
{
    char line[ MM_MAX_LINE_LENGTH ];
    char banner[ MM_MAX_TOKEN_LENGTH ];
    char mtx[ MM_MAX_TOKEN_LENGTH ];
    char crd[ MM_MAX_TOKEN_LENGTH ];
    char data_type[ MM_MAX_TOKEN_LENGTH ];
    char storage_scheme[ MM_MAX_TOKEN_LENGTH ];
    char *p;

    mm_clear_typecode( Typecode );

    if( fgets( line, MM_MAX_LINE_LENGTH, infile ) == NULL )
        return MM_PREMATURE_EOF;

    if( sscanf( line, "%s %s %s %s %s", banner, mtx, crd, data_type,
        storage_scheme ) != 5 )
        return MM_PREMATURE_EOF;

    for( p = mtx; *p != '\0'; *p = tolower( *p ), p++ );  /* convert to lower case */
    for( p = crd; *p != '\0'; *p = tolower( *p ), p++ );
    for( p = data_type; *p != '\0'; *p = tolower( *p ), p++ );
    for( p = storage_scheme; *p != '\0'; *p = tolower( *p ), p++ );

    /* check for banner */
    if( strncmp( banner, MatrixMarketBanner, strlen( MatrixMarketBanner ) ) != 0 )
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if( strcmp( mtx, MM_MTX_STR ) != 0 )
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix( Typecode );

    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */
    if( strcmp( crd, MM_SPARSE_STR ) == 0 )
        mm_set_sparse( Typecode );
    else if( strcmp( crd, MM_DENSE_STR ) == 0 )
        mm_set_dense( Typecode );
    else
        return MM_UNSUPPORTED_TYPE;


    /* third field */

    if( strcmp( data_type, MM_REAL_STR ) == 0 )
        mm_set_real( Typecode );
    else
        if( strcmp( data_type, MM_COMPLEX_STR ) == 0 )
            mm_set_complex( Typecode );
        else
            if( strcmp( data_type, MM_PATTERN_STR ) == 0 )
                mm_set_pattern( Typecode );
            else
                if( strcmp( data_type, MM_INT_STR ) == 0 )
                    mm_set_integer( Typecode );
                else
                    return MM_UNSUPPORTED_TYPE;


    /* fourth field */

    if( strcmp( storage_scheme, MM_GENERAL_STR ) == 0 )
        mm_set_general( Typecode );
    else
        if( strcmp( storage_scheme, MM_SYMM_STR ) == 0 )
            mm_set_symmetric( Typecode );
        else
            if( strcmp( storage_scheme, MM_HERM_STR ) == 0 )
                mm_set_hermitian( Typecode );
            else
                if( strcmp( storage_scheme, MM_SKEW_STR ) == 0 )
                    mm_set_skew( Typecode );
                else
                    return MM_UNSUPPORTED_TYPE;

    return 0;

}

template<typename FloatType>
int MatrixMarketReader<FloatType>::MMReadMtxCrdSize( FILE *infile )
{
    char line[ MM_MAX_LINE_LENGTH ];
    int num_items_read;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if( fgets( line, MM_MAX_LINE_LENGTH, infile ) == NULL )
            return MM_PREMATURE_EOF;
    } while( line[ 0 ] == '%' );

    /* line[] is either blank or has M,N, nz */
    if( sscanf( line, "%d %d %d", &nRows, &nCols, &nNZ ) == 3 )
        return 0;
    else
        do
        {
            num_items_read = fscanf( infile, "%d %d %d", &nRows, &nCols, &nNZ );
            if( num_items_read == EOF ) return MM_PREMATURE_EOF;
        } while( num_items_read != 3 );

    return 0;
}
#endif
