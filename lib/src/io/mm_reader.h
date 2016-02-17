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
    Coordinate<FloatType> *unsym_coords;

public:
    MatrixMarketReader( ): nNZ( 0 ), nRows( 0 ), nCols( 0 ), isSymmetric( 0 ), isDoubleMem( 0 )
    {
        for( auto c : Typecode )
            c = '\0';

        unsym_coords = NULL;
    }

    bool MMReadHeader( FILE* infile );
    bool MMReadHeader( const std::string& filename );
    bool MMReadFormat( const std::string& _filename, bool read_explicit_zeroes );
    int MMReadBanner( FILE* infile );
    int MMReadMtxCrdSize( FILE* infile );
    void MMGenerateCOOFromFile( FILE* infile, bool read_explicit_zeroes );

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

    Coordinate<FloatType> *GetUnsymCoordinates( )
    {
        return unsym_coords;
    }

    ~MatrixMarketReader( )
    {
        delete[ ] unsym_coords;
    }
};

// Class definition

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadHeader( FILE* mm_file )
{
    int status = MMReadBanner( mm_file );
    if( status != 0 )
    {
        printf( "Error Reading Banner in Matrix-Market File !\n" );
        return 1;
    }

    if( !mm_is_coordinate( Typecode ) )
    {
        printf( "Handling only coordinate format\n" ); return( 1 );
    }

    if( mm_is_complex( Typecode ) )
    {
        printf( "Error: cannot handle complex format\n" );
        return ( 1 );
    }

    if( mm_is_symmetric( Typecode ) )
        isSymmetric = 1;

    status = MMReadMtxCrdSize( mm_file );
    if( status != 0 )
    {
        printf( "Error reading Matrix Market crd_size %d\n", status );
        return( 1 );
    }

    return 0;
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadHeader( const std::string &filename )
{
    FILE *mm_file = ::fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        printf( "Cannot Open Matrix-Market File !\n" );
        return 1;
    }

    if ( MMReadHeader( mm_file ) )
    {
        printf ("Matrix not supported !\n");
        return 2;
    }

    // If symmetric MM stored file, double the reported size
    if( mm_is_symmetric( Typecode ) )
        nNZ <<= 1;

    ::fclose( mm_file );

    std::clog << "Matrix: " << filename << " [nRow: " << GetNumRows( ) << "] [nCol: " << GetNumCols( ) << "] [nNZ: " << GetNumNonZeroes( ) << "]" << std::endl;

    return 0;
}

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

template<typename FloatType>
void FillCoordData( char Typecode[ ],
                    Coordinate<FloatType> *unsym_coords,
                    int &unsym_actual_nnz,
                    int ir,
                    int ic,
                    FloatType value )
{
    if( mm_is_symmetric( Typecode ) )
    {
        unsym_coords[ unsym_actual_nnz ].x = ir - 1;
        unsym_coords[ unsym_actual_nnz ].y = ic - 1;
        unsym_coords[ unsym_actual_nnz++ ].val = value;

        if( unsym_coords[ unsym_actual_nnz - 1 ].x != unsym_coords[ unsym_actual_nnz - 1 ].y )
        {
            unsym_coords[ unsym_actual_nnz ].x = unsym_coords[ unsym_actual_nnz - 1 ].y;
            unsym_coords[ unsym_actual_nnz ].y = unsym_coords[ unsym_actual_nnz - 1 ].x;
            unsym_coords[ unsym_actual_nnz ].val = unsym_coords[ unsym_actual_nnz - 1 ].val;
            unsym_actual_nnz++;
        }
    }
    else
    {
        unsym_coords[ unsym_actual_nnz ].x = ir - 1;
        unsym_coords[ unsym_actual_nnz ].y = ic - 1;
        unsym_coords[ unsym_actual_nnz++ ].val = value;
    }
}

template<typename FloatType>
void MatrixMarketReader<FloatType>::MMGenerateCOOFromFile( FILE *infile, bool read_explicit_zeroes )
{
    int unsym_actual_nnz = 0;
    FloatType value;
    int ir, ic;

    const int exp_zeroes = read_explicit_zeroes;

    //silence warnings from fscanf (-Wunused-result)
    int rv = 0;

    for( int i = 0; i < nNZ; i++ )
    {
        if( mm_is_real( Typecode ) )
        {
            if( typeid( FloatType ) == typeid( float ) )
              rv = fscanf( infile, "%d %d %f\n", &ir, &ic, (float*)( &value ) );
            else if( typeid( FloatType ) == typeid( double ) )
              rv = fscanf( infile, "%d %d %lf\n", &ir, &ic, (double*)( &value ) );

            if( exp_zeroes == 0 && value == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, value );
        }
        else if( mm_is_integer( Typecode ) )
        {
            if(typeid(FloatType) == typeid(float))
               rv = fscanf(infile, "%d %d %f\n", &ir, &ic, (float*)( &value ) );
            else if(typeid(FloatType) == typeid(double))
               rv = fscanf(infile, "%d %d %lf\n", &ir, &ic, (double*)( &value ) );

            if( exp_zeroes == 0 && value == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, value );

        }
        else if( mm_is_pattern( Typecode ) )
        {
            rv = fscanf( infile, "%d %d", &ir, &ic );
            value = static_cast<FloatType>( MAX_RAND_VAL * ( rand( ) / ( RAND_MAX + 1.0 ) ) );

            if( exp_zeroes == 0 && value == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, value );
        }
    }
    nNZ = unsym_actual_nnz;
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadFormat( const std::string &filename, bool read_explicit_zeroes )
{
    FILE *mm_file = ::fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        printf( "Cannot Open Matrix-Market File !\n" );
        return 1;
    }

    if ( MMReadHeader( mm_file ) )
    {
        printf ("Matrix not supported !\n");
        return 2;
    }

    if( mm_is_symmetric( Typecode ) )
        unsym_coords = new Coordinate<FloatType>[ 2 * nNZ ];
    else
        unsym_coords = new Coordinate<FloatType>[ nNZ ];

    MMGenerateCOOFromFile( mm_file, read_explicit_zeroes );
    ::fclose( mm_file );

    return 0;
}
#endif
