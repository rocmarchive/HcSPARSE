#ifndef _HC_SPARSE_STRUCT_H_
#define _HC_SPARSE_STRUCT_H_

#include <iostream>
#include <amp.h>

typedef enum _hcdenseMajor
{
    rowMajor = 1,
    columnMajor
} hcdenseMajor;

/*! \brief Structure to encapsulate scalar data to hcsparse API
 */
typedef struct hcsparseScalar_
{
    Concurrency::array_view<float> *value;
    long offValue;
    void clear( )
    {
        value = nullptr;
    }

    long offset () const
    {
        return 0;
    }
} hcsparseScalar;

/*! \brief Structure to encapsulate dense vector data to hcsparse API
 */
typedef struct hcdenseVector_
{
    int num_values;
    Concurrency::array_view<float> *values;
    long offValues;
    void clear( )
    {
        num_values = 0;
        values = nullptr;
    }

    long offset () const
    {
        return 0;
    }
} hcdenseVector;

/*! \brief Structure to encapsulate sparse matrix data encoded in CSR
 * form to hcsparse API
 * \note The indices stored are 0-based
 */
typedef struct hcsparseCsrMatrix_
{
    /** @name CSR matrix data */
    /**@{*/
    int num_rows;  /*!< Number of rows this matrix has if viewed as dense */
    int num_cols;  /*!< Number of columns this matrix has if viewed as dense */
    int num_nonzeros;  /*!< Number of values in matrix that are non-zero */
    /**@}*/

    /** @name OpenCL state */
    /**@{*/
    Concurrency::array_view<float> *values;  /*!< non-zero values in sparse matrix of size num_nonzeros */
    Concurrency::array_view<float> *colIndices;  /*!< column index for corresponding value of size num_nonzeros */
    Concurrency::array_view<float> *rowOffsets;  /*!< Invariant: rowOffsets[i+1]-rowOffsets[i] = number of values in row i */
    Concurrency::array_view<float> *rowBlocks;  /*!< Meta-data used for csr-adaptive algorithm; can be NULL */
    /**@}*/

    /** @name Buffer offsets */

    long offValues;
    long offColInd;
    long offRowOff;
    long offRowBlocks;
    /**@}*/

    size_t rowBlockSize;  /*!< Size of array used by the rowBlocks handle */
    void clear( )
    {
        num_rows = num_cols = num_nonzeros = 0;
        values = nullptr;
        colIndices = rowOffsets = rowBlocks = nullptr;
        rowBlockSize = 0;
    }

    uint nnz_per_row() const
    {
        return num_nonzeros / num_rows;
    }

    long valOffset () const
    {
        return 0;
    }

    long colIndOffset () const
    {
        return 0;
    }

    long rowOffOffset () const
    {
        return 0;
    }

    long rowBlocksOffset( ) const
    {
        return 0;
    }
} hcsparseCsrMatrix;

/*! \brief Structure to encapsulate sparse matrix data encoded in COO
 * form to hcsparse API
 * \note The indices stored are 0-based
 */
typedef struct hcsparseCooMatrix_
{
    /** @name COO matrix data */
    /**@{*/
    int num_rows;  /*!< Number of rows this matrix has if viewed as dense */
    int num_cols;  /*!< Number of columns this matrix has if viewed as dense */
    int num_nonzeros;  /*!< Number of values in matrix that are non-zero */
    /**@}*/

    /** @name OpenCL state */
    /**@{*/
    Concurrency::array_view<float> *values;  /*!< CSR non-zero values of size num_nonzeros */
    Concurrency::array_view<float> *colIndices;  /*!< column index for corresponding element; array size num_nonzeros */
    Concurrency::array_view<float> *rowIndices;  /*!< row index for corresponding element; array size num_nonzeros */
    /**@}*/

    /** @name Buffer offsets */

    long offValues;
    long offColInd;
    long offRowInd;
    /**@}*/
    void clear( )
    {
        num_rows = num_cols = num_nonzeros = 0;
        values = colIndices = rowIndices = nullptr;
    }

    uint nnz_per_row( ) const
    {
        return num_nonzeros / num_rows;
    }

    long valOffset( ) const
    {
        return 0;
    }

    long colIndOffset( ) const
    {
        return 0;
    }

    long rowOffOffset( ) const
    {
        return 0;
    }
} hcsparseCooMatrix;

/*! \brief Structure to encapsulate dense matrix data to hcsparse API
 */
typedef struct hcdenseMatrix_
{
    /** @name Dense matrix data */
    /**@{*/
    size_t num_rows;  /*!< Number of rows */
    size_t num_cols;  /*!< Number of columns */
    size_t lead_dim;  /*! Stride to the next row or column, in units of elements */
    hcdenseMajor major;  /*! Memory layout for dense matrix */
    /**@}*/

    Concurrency::array_view<float> *values;  /*!< Array of matrix values */

    long offValues;

    void clear( )
    {
        num_rows = num_cols = lead_dim = 0;
        major = rowMajor;
        values = nullptr;
    }
} hcdenseMatrix;

#endif
