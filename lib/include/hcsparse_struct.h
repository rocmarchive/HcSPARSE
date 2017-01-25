#ifndef _HC_SPARSE_STRUCT_H_
#define _HC_SPARSE_STRUCT_H_

#include <iostream>
#include <hc.hpp>
#include <hc_math.hpp>
#include <hc_am.hpp>
using namespace hc;
using namespace hc::precise_math;

/* Class which implements the hcsparse library */
class hcsparseLibrary
{
public:

    // Add current Accerator field
    hc::accelerator currentAccl;

    // Filed to check if library is initialized
    bool initialized = false;

    hc::accelerator_view currentAcclView;

    // StreamInfo
    void* currentStream = NULL;

    // for NV(32) for AMD(64)
    size_t wavefront_size = 0;
    size_t max_wg_size = 0;

    // Should we attempt to perform compensated summation?
    bool extended_precision = false;

    // Does our device have double precision support?
    bool dpfp_support = false;

    // current device max compute units;
    uint max_compute_units = 0;

    // Constructor to initialize the library with the given accelerator view
    hcsparseLibrary(hc::accelerator_view *av)
        : currentAcclView(*av), currentAccl(av->get_accelerator()),
        wavefront_size(0), max_wg_size(0), extended_precision(false),
        dpfp_support(false), max_compute_units(0)

    {
      std::vector<accelerator> accs = accelerator::get_all();
      for (int i=0;i<accs.size();i++) {
        if (accs[i] == this->currentAccl) {
          this-> initialized = true;
          break;
        }
      }
      assert(this->initialized);
    }

    // Destructor
    ~hcsparseLibrary()
    {
       // Deinitialize the library
       this->initialized = false;
    }

};

template<typename T>
struct Coordinate
{
    int x;
    int y;
    T val;
};

typedef enum hcsparseStatus_
{
    /** @name Inherited OpenCL codes */
    /**@{*/
    hcsparseSuccess = 1,
    hcsparseInvalid = 0,
} hcsparseStatus;

typedef enum _hcdenseMajor
{
    rowMajor = 1,
    columnMajor
} hcdenseMajor;

/*! \brief Enumeration to *control the verbosity of the sparse iterative
 * solver routines.  VERBOSE will print helpful diagnostic messages to
 * console
 *
 * \ingroup SOLVER
 */

typedef enum _print_mode
{
    QUIET = 0,
    NORMAL,
    VERBOSE
} PRINT_MODE;

/*! \brief Enumeration to select the preconditioning algorithm used to precondition
 * the sparse data before the iterative solver execution phase
 *
 * \ingroup SOLVER
 */
typedef enum _precond
{
    NOPRECOND = 0,
    DIAGONAL
} PRECONDITIONER;

/*! \brief Structure to encapsulate scalar data to hcsparse API
 */
typedef struct hcsparseScalar_
{
    void* value;
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
    void* values;
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

struct hcsparseControl_
{
    hc::accelerator_view accl_view;

    // for NV(32) for AMD(64)
    size_t wavefront_size;
    size_t max_wg_size;

    // Should we attempt to perform compensated summation?
    bool extended_precision;

    // Does our device have double precision support?
    bool dpfp_support;

    // current device max compute units;
    uint max_compute_units;

    hcsparseControl_( hc::accelerator_view &accl_view )
        : accl_view( accl_view ), wavefront_size( 0 ),
        max_wg_size( 0 ), extended_precision(false),
        dpfp_support(false), max_compute_units( 0 )
    {}

};
typedef struct hcsparseControl_ hcsparseControl;

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
    void *values;  /*!< non-zero values in sparse matrix of size num_nonzeros */
    void *colIndices;  /*!< column index for corresponding value of size num_nonzeros */
    void *rowOffsets;  /*!< Invariant: rowOffsets[i+1]-rowOffsets[i] = number of values in row i */
    void *rowBlocks;  /*!< Meta-data used for csr-adaptive algorithm; can be NULL */
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
    void *values;  /*!< CSR non-zero values of size num_nonzeros */
    void *colIndices;  /*!< column index for corresponding element; array size num_nonzeros */
    void *rowIndices;  /*!< row index for corresponding element; array size num_nonzeros */
    /**@}*/

    /** @name Buffer offsets */

    long offValues;
    long offColInd;
    long offRowInd;
    /**@}*/
    void clear( )
    {
        num_rows = num_cols = num_nonzeros = 0;
        values = nullptr;
        colIndices = rowIndices = nullptr;
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

    void*values;  /*!< Array of matrix values */

    long offValues;

    void clear( )
    {
        num_rows = num_cols = lead_dim = 0;
        major = rowMajor;
        values = nullptr;
    }
} hcdenseMatrix;

typedef struct _solverControl
{

    _solverControl() : nIters(0), maxIters(0), preconditioner(NOPRECOND),
        relativeTolerance(0.0), absoluteTolerance(0.0),
        initialResidual(0), currentResidual(0), printMode(VERBOSE)
    {

    }

    bool finished(const double residuum)
    {
        return converged(residuum) || nIters >= maxIters;
    }

    bool converged(const double residuum)
    {
        currentResidual = residuum;
        if(residuum <= relativeTolerance ||
           residuum <= absoluteTolerance * initialResidual)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void print()
    {
        if (printMode == VERBOSE)
        {
            std::cout << "Iteration: " << nIters
                      << " Residuum: " << currentResidual
                      << std::endl;
        }
    }

    std::string printPreconditioner()
    {

        switch(preconditioner)
        {
        case NOPRECOND:
            return "No preconditioner";
        case DIAGONAL:
            return "Diagonal";
        }
    }

    void printSummary(hcsparseStatus status)
    {
        if (printMode == VERBOSE || printMode == NORMAL)
        {
            std::cout << "Solver constraints: \n"
                      << "\trelative tolerance = " << relativeTolerance
                      << "\n\tabsolute tolerance = " << absoluteTolerance
                      << "\n\tmax iterations = " << maxIters
                      << "\n\tPreconditioner: " << printPreconditioner()
                      << std::endl;

            std::cout << "Solver finished calculations with status "
                      << status << std::endl;

            std::cout << "\tfinal residual = " << currentResidual
                      << "\titerations = " << nIters
                      << std::endl;
        }
    }

    // current solver iteration;
    int nIters;

    // maximum solver iterations;
    int maxIters;

    // preconditioner type
    PRECONDITIONER preconditioner;

    // required relative tolerance
    double relativeTolerance;

    // required absolute tolerance
    double absoluteTolerance;

    double initialResidual;

    double currentResidual;

    PRINT_MODE printMode;
} hcsparseSolverControl;

#endif
