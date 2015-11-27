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

#endif
