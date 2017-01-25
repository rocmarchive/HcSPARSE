#include "hcsparse.h"
#include "hc_am.hpp"
#include "blas2/csrmv.h"
#include "blas3/csrmm.h"
#include "blas3/hcsparse-spm-spm.h"
#include "blas1/hcdense-scale.h"
#include "blas1/hcdense-axpby.h"
#include "blas1/hcdense-axpy.h"
#include "blas1/hcdense-reduce.h"
#include "blas1/reduce-operators.h"
#include "blas1/hcdense-nrm1.h"
#include "blas1/hcdense-nrm2.h"
#include "blas1/hcdense-dot.h"
#include "blas1/elementwise-transform.h"
#include "io/mm_reader.h"
#include "blas2/csr_meta.h"
#include "solvers/preconditioners/preconditioner.h"
#include "solvers/preconditioners/diagonal.h"
#include "solvers/preconditioners/void.h"
#include "solvers/solver-control.h"
#include "solvers/biconjugate-gradients-stabilized.h"
#include "solvers/conjugate-gradients.h"
#include "transform/scan.h"
#include "transform/reduce-by-key.h"
#include "transform/conversion-utils.h"
#include "transform/hcsparse-coo2csr.h"
#include "transform/hcsparse-csr2coo.h"
#include "transform/hcsparse-csr2dense.h"
#include "transform/hcsparse-dense2csr.h"

int hcsparseInitialized = 0;

// hcsparse Helper functions 

// 1. hcsparseCreate()

// This function initializes the HCSPARSE library and creates a handle to an opaque structure
// holding the HCSPARSE library context.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_ALLOC_FAILED       the resources could not be allocated  

hcsparseStatus_t hcsparseCreate(hcsparseHandle_t *handle, hc::accelerator_view *accl_view) {
  if (handle == NULL) {
    handle = new hcsparseHandle_t();
  }

  *handle = new hcsparseLibrary(accl_view);

  if(*handle == NULL) {
    return HCSPARSE_STATUS_ALLOC_FAILED;
  }
  return HCSPARSE_STATUS_SUCCESS;
}

// 2. hcsparseDestory()

// This function releases hardware resources used by the HCSPARSE library.
// This function is usually the last call with a particular handle to the HCSPARSE library.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            the shut down succeeded
// HCSPARSE_STATUS_NOT_INITIALIZED    the library was not initialized

hcsparseStatus_t hcsparseDestroy(hcsparseHandle_t *handle){
  if(handle == nullptr || *handle == nullptr)
    return HCSPARSE_STATUS_NOT_INITIALIZED;
  delete *handle;
  *handle = nullptr;
  handle = nullptr;
  return HCSPARSE_STATUS_SUCCESS;
}

// 3. hcsparseSetAcclView()

//This function sets the hcSPARSE library stream, which will be used to execute all subsequent calls to the hcSPARSE library functions. If the hcSPARSE library stream is not set, all kernels use the defaultNULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the hcSPARSE library stream back to NULL.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS         :the stream was set successfully
// HCSPARSE_STATUS_NOT_INITIALIZED :the library was not initialized
hcsparseStatus_t hcsparseSetAcclView(hcsparseHandle_t handle, hc::accelerator_view accl_view, void* stream) {
  if (handle == nullptr || handle->initialized == false) {
    return HCSPARSE_STATUS_NOT_INITIALIZED;    
  }
  handle->currentAcclView = accl_view;
  handle->currentStream = stream;
  return HCSPARSE_STATUS_SUCCESS;
}

// 4. hcsparseGetAcclView()

// This function gets the hcSPARSE library stream, which is being used to execute all calls to the hcSPARSE library functions. If the hcSPARSE library stream is not set, all kernels use the defaultNULL stream.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS : the stream was returned successfully
// HCSPARSE_STATUS_NOT_INITIALIZED : the library was not initialized

hcsparseStatus_t  hcsparseGetAcclView(hcsparseHandle_t handle, hc::accelerator_view **accl_view, void** stream) {
  if (handle == nullptr) {
    return HCSPARSE_STATUS_NOT_INITIALIZED;    
  }
  *accl_view = &handle->currentAcclView;
  stream = &(handle->currentStream);
  return HCSPARSE_STATUS_SUCCESS;
}

// 5. hcsparseCreateMatDescr()

// This function initializes the matrix descriptor. It sets the fields MatrixType
// and IndexBase to the default values HCSPARSE_MATRIX_TYPE_GENERAL and
// HCSPARSE_INDEX_BASE_ZERO, respectively, while leaving other fields uninitialized.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_ALLOC_FAILED       the resources could not be allocated  
hcsparseStatus_t
hcsparseCreateMatDescr(hcsparseMatDescr_t *descrA) {
  if (descrA == nullptr) {
    descrA = (hcsparseMatDescr_t*)malloc(sizeof(hcsparseMatDescr_t));
    if (descrA == NULL)
      return HCSPARSE_STATUS_ALLOC_FAILED;
  }

  descrA->MatrixType = HCSPARSE_MATRIX_TYPE_GENERAL;
  descrA->IndexBase = HCSPARSE_INDEX_BASE_ZERO;
  return HCSPARSE_STATUS_SUCCESS;
}

// 6. hcsparseDestroyMatDescr()

// This function releases the memory allocated for the matrix descriptor.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded

hcsparseStatus_t
hcsparseDestroyMatDescr(hcsparseMatDescr_t *descrA) {
  if (descrA != NULL) {
    descrA = NULL;
  }
  return HCSPARSE_STATUS_SUCCESS;
}


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
hcsparseInitScalar (hcsparseScalar* scalar)
{
    scalar->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitVector (hcdenseVector* vec)
{
    vec->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitCooMatrix (hcsparseCooMatrix* cooMatx)
{
    cooMatx->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcsparseInitCsrMatrix (hcsparseCsrMatrix* csrMatx)
{
    csrMatx->clear( );

    return hcsparseSuccess;
};

hcsparseStatus
hcdenseInitMatrix (hcdenseMatrix* denseMatx)
{
    denseMatx->clear();

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseScsrmv (const hcsparseScalar* alpha,
                const hcsparseCsrMatrix* matx,
                const hcdenseVector* x,
                const hcsparseScalar* beta,
                hcdenseVector* y,
                hcsparseControl* control)
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
hcsparseDcsrmv (const hcsparseScalar* alpha,
                const hcsparseCsrMatrix* matx,
                const hcdenseVector* x,
                const hcsparseScalar* beta,
                hcdenseVector* y,
                hcsparseControl* control)
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
hcsparseScsrmm (const hcsparseScalar* alpha,
                const hcsparseCsrMatrix* sparseCsrA,
                const hcdenseMatrix* denseB,
                const hcsparseScalar* beta,
                hcdenseMatrix* denseC,
                hcsparseControl *control)
{
    if( !hcsparseInitialized )
    {
        return hcsparseInvalid;
    }

    if( denseB->values == nullptr || denseC->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csrmm<float> ( alpha, sparseCsrA, denseB, beta, denseC, control );
}

hcsparseStatus
hcsparseDcsrmm (const hcsparseScalar* alpha,
                const hcsparseCsrMatrix* sparseCsrA,
                const hcdenseMatrix* denseB,
                const hcsparseScalar* beta,
                hcdenseMatrix* denseC,
                hcsparseControl *control)
{
    if( !hcsparseInitialized )
    {
        return hcsparseInvalid;
    }

    if( denseB->values == nullptr || denseC->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csrmm<double> ( alpha, sparseCsrA, denseB, beta, denseC, control );
}

hcsparseStatus
hcdenseSscale (hcdenseVector* r,
               const hcsparseScalar* alpha,
               const hcdenseVector* y,
               hcsparseControl* control)
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
hcdenseDscale (hcdenseVector* r,
               const hcsparseScalar* alpha,
               const hcdenseVector* y,
               hcsparseControl* control)
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

hcsparseStatus
hcdenseSaxpy (hcdenseVector* r,
              const hcsparseScalar* alpha,
              const hcdenseVector* x,
              const hcdenseVector* y,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr || x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return axpy<float> (r, alpha, x, y,  control);
}

hcsparseStatus
hcdenseDaxpy (hcdenseVector* r,
              const hcsparseScalar* alpha,
              const hcdenseVector* x,
              const hcdenseVector* y,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr || x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return axpy<double> (r, alpha, x, y,  control);
}

hcsparseStatus
hcdenseSaxpby (hcdenseVector* r,
               const hcsparseScalar* alpha,
               const hcdenseVector* x,
               const hcsparseScalar* beta,
               const hcdenseVector* y,
               hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr || x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return axpby<float> (r, alpha, x, beta, y,  control);
}

hcsparseStatus
hcdenseDaxpby (hcdenseVector* r,
               const hcsparseScalar* alpha,
               const hcdenseVector* x,
               const hcsparseScalar* beta,
               const hcdenseVector* y,
               hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || y->values == nullptr || x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return axpby<double> (r, alpha, x, beta, y,  control);
}

hcsparseStatus
hcdenseIreduce (hcsparseScalar* s,
                const hcdenseVector* x,
                hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return reduce<int, RO_PLUS>(s, x, control);
}

hcsparseStatus
hcdenseSreduce (hcsparseScalar* s,
                const hcdenseVector* x,
                hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return reduce<float, RO_PLUS>(s, x, control);
}

hcsparseStatus
hcdenseDreduce (hcsparseScalar* s,
                const hcdenseVector* x,
                hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return reduce<double, RO_PLUS>(s, x, control);
}

hcsparseStatus
hcdenseSadd (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<float, EW_PLUS> (r, x, y, control);
}

hcsparseStatus
hcdenseDadd (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<double, EW_PLUS> (r, x, y, control);
}

hcsparseStatus
hcdenseSsub (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<float, EW_MINUS> (r, x, y, control);
}

hcsparseStatus
hcdenseDsub (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<double, EW_MINUS> (r, x, y, control);
}

hcsparseStatus
hcdenseSmul (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<float, EW_MULTIPLY> (r, x, y, control);
}

hcsparseStatus
hcdenseDmul (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<double, EW_MULTIPLY> (r, x, y, control);
}

hcsparseStatus
hcdenseSdiv (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<float, EW_DIV> (r, x, y, control);
}

hcsparseStatus
hcdenseDdiv (hcdenseVector* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (r->values == nullptr || x->values == nullptr || y->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return elementwise_transform<double, EW_DIV> (r, x, y, control);
}

hcsparseStatus
hcdenseSnrm1 (hcsparseScalar* s,
              const hcdenseVector* x,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return Norm1<float>(s, x, control);
}

hcsparseStatus
hcdenseDnrm1 (hcsparseScalar* s,
              const hcdenseVector* x,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return Norm1<double>(s, x, control);
}

hcsparseStatus
hcdenseSnrm2 (hcsparseScalar* s,
              const hcdenseVector* x,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return Norm2<float>(s, x, control);
}

hcsparseStatus
hcdenseDnrm2 (hcsparseScalar* s,
              const hcdenseVector* x,
              hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return Norm2<double>(s, x, control);
}

hcsparseStatus
hcdenseSdot (hcsparseScalar* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
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

    return dot<float>(r, x, y, control);
}

hcsparseStatus
hcdenseDdot (hcsparseScalar* r,
             const hcdenseVector* x,
             const hcdenseVector* y,
             hcsparseControl* control)
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

    return dot<double>(r, x, y, control);
}

// This function reads the file header at the given filepath, and returns the size
// of the sparse matrix in the hcsparseCooMatrix parameter.
// Post-condition: clears hcsparseCooMatrix, then sets pCooMatx->m, pCooMatx->n
// pCooMatx->nnz
hcsparseStatus
hcsparseHeaderfromFile (int* nnz, int* row, int* col, const char* filePath)
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

template<typename T>
bool CoordinateCompare (const Coordinate<T> &c1, const Coordinate<T> &c2)
{
    if( c1.x != c2.x )
        return ( c1.x < c2.x );
    else
        return ( c1.y < c2.y );
}

// This function reads the file at the given filepath, and returns the sparse
// matrix in the COO struct.  All matrix data is written to device memory
// Pre-condition: This function assumes that the device memory buffers have been
// pre-allocated by the caller
hcsparseStatus
hcsparseSCooMatrixfromFile (hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes)
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
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

    Coordinate<float>* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + cooMatx->num_nonzeros, CoordinateCompare< float > );

    float *values = (float*) calloc(cooMatx->num_nonzeros, sizeof(float));
    int *rowIndices = (int*) calloc(cooMatx->num_nonzeros, sizeof(int));
    int *colIndices = (int*) calloc(cooMatx->num_nonzeros, sizeof(int));

    for( int c = 0; c < cooMatx->num_nonzeros; ++c )
    {
        rowIndices[ c ] = coords[c].x;
        colIndices[ c ] = coords[c].y;
        values[ c ] = coords[c].val;
    }

    control->accl_view.copy(values, cooMatx->values, cooMatx->num_nonzeros * sizeof(float));
    control->accl_view.copy(rowIndices, cooMatx->rowIndices, cooMatx->num_nonzeros * sizeof(int));
    control->accl_view.copy(colIndices, cooMatx->colIndices, cooMatx->num_nonzeros * sizeof(int));

    free(values);
    free(rowIndices);
    free(colIndices);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseDCooMatrixfromFile (hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes)
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
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

    Coordinate<double>* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + cooMatx->num_nonzeros, CoordinateCompare<double> );

    double *values = (double*) calloc(cooMatx->num_nonzeros, sizeof(double));
    int *rowIndices = (int*) calloc(cooMatx->num_nonzeros, sizeof(int));
    int *colIndices = (int*) calloc(cooMatx->num_nonzeros, sizeof(int));

    for( int c = 0; c < cooMatx->num_nonzeros; ++c )
    {
        rowIndices[ c ] = coords[c].x;
        colIndices[ c ] = coords[c].y;
        values[ c ] = coords[c].val;
    }

    control->accl_view.copy(values, cooMatx->values, cooMatx->num_nonzeros * sizeof(double));
    control->accl_view.copy(rowIndices, cooMatx->rowIndices, cooMatx->num_nonzeros * sizeof(int));
    control->accl_view.copy(colIndices, cooMatx->colIndices, cooMatx->num_nonzeros * sizeof(int));

    free(values);
    free(rowIndices);
    free(colIndices);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseSCsrMatrixfromFile (hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes)
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

    Coordinate<float>* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + csrMatx->num_nonzeros, CoordinateCompare< float > );

    float *values = (float*)calloc(csrMatx->num_nonzeros, sizeof(float));
    int *rowOffsets = (int*)calloc((csrMatx->num_rows)+1, sizeof(int));
    int *colIndices = (int*)calloc(csrMatx->num_nonzeros, sizeof(int));

    int current_row = 0;
    rowOffsets[ 0 ] = 0;
    for( int i = 0; i < csrMatx->num_nonzeros; i++ )
    {
        colIndices[ i ] = coords[i].y;
        values[ i ] = coords[i].val;

        while( coords[i].x >= current_row )
            rowOffsets[ current_row++ ] = i;
    }
    rowOffsets[ current_row ] = csrMatx->num_nonzeros;
    while( current_row <= csrMatx->num_rows )
        rowOffsets[ current_row++ ] = csrMatx->num_nonzeros;

    control->accl_view.copy(values, csrMatx->values, sizeof(float) * csrMatx->num_nonzeros);
    control->accl_view.copy(rowOffsets, csrMatx->rowOffsets, sizeof(int) * (csrMatx->num_rows+1));
    control->accl_view.copy(colIndices, csrMatx->colIndices, sizeof(int) * csrMatx->num_nonzeros);

    free(values);
    free(rowOffsets);
    free(colIndices);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseDCsrMatrixfromFile (hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes)
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

    Coordinate<double>* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + csrMatx->num_nonzeros, CoordinateCompare<double> );

    double *values = (double*)calloc(csrMatx->num_nonzeros, sizeof(double));
    int *rowOffsets = (int*)calloc(csrMatx->num_rows+1, sizeof(int));
    int *colIndices = (int*)calloc(csrMatx->num_nonzeros, sizeof(int));

    int current_row = 0;
    rowOffsets[ 0 ] = 0;
    for( int i = 0; i < csrMatx->num_nonzeros; i++ )
    {
        colIndices[ i ] = coords[i].y;
        values[ i ] = coords[i].val;

        while( coords[i].x >= current_row )
            rowOffsets[ current_row++ ] = i;
    }
    rowOffsets[ current_row ] = csrMatx->num_nonzeros;
    while( current_row <= csrMatx->num_rows )
        rowOffsets[ current_row++ ] = csrMatx->num_nonzeros;

    control->accl_view.copy(values, csrMatx->values, sizeof(double) * csrMatx->num_nonzeros);
    control->accl_view.copy(rowOffsets, csrMatx->rowOffsets, sizeof(int) * (csrMatx->num_rows+1));
    control->accl_view.copy(colIndices, csrMatx->colIndices, sizeof(int) * csrMatx->num_nonzeros);

    free(values);
    free(rowOffsets);
    free(colIndices);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseCsrMetaSize (hcsparseCsrMatrix* csrMatx, hcsparseControl *control)
{
    int *rCsrRowOffsets = (int*)calloc(csrMatx->num_rows+1, sizeof(int));
    control->accl_view.copy(csrMatx->rowOffsets, rCsrRowOffsets, sizeof(int) * (csrMatx->num_rows+1));

    csrMatx->rowBlockSize = ComputeRowBlocksSize( rCsrRowOffsets, csrMatx->num_rows, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR );

    control->accl_view.copy(rCsrRowOffsets, csrMatx->rowOffsets, sizeof(int) * (csrMatx->num_rows+1));

    free(rCsrRowOffsets);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseCsrMetaCompute (hcsparseCsrMatrix* csrMatx, hcsparseControl *control)
{
    // Check to ensure nRows can fit in 32 bits
    if( static_cast<ulong>( csrMatx->num_rows ) > static_cast<ulong>( std::pow( 2, ( 64 - ROWBITS ) ) ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present ((64-WG_BITS) bits) !" );
        return hcsparseInvalid;
    }

    int *rCsrRowOffsets = (int*)calloc(csrMatx->num_rows+1, sizeof(int));
    ulong *rRowBlocks = (ulong*)calloc(csrMatx->num_nonzeros, sizeof(ulong));

    control->accl_view.copy(csrMatx->rowOffsets, rCsrRowOffsets, sizeof(int) * (csrMatx->num_rows+1));
    control->accl_view.copy(csrMatx->rowBlocks, rRowBlocks, sizeof(ulong) * csrMatx->num_nonzeros);

    ComputeRowBlocks(rRowBlocks, csrMatx->rowBlockSize, rCsrRowOffsets, csrMatx->num_rows, BLOCKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, true );

    control->accl_view.copy(rCsrRowOffsets, csrMatx->rowOffsets, sizeof(int) * (csrMatx->num_rows+1));
    control->accl_view.copy(rRowBlocks, csrMatx->rowBlocks, sizeof(ulong) * csrMatx->num_nonzeros);

    free(rCsrRowOffsets);
    free(rRowBlocks);

    return hcsparseSuccess;
}

hcsparseStatus
hcsparseScsrbicgStab (hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                      hcsparseSolverControl *solverControl, hcsparseControl *control)
{
    using T = float;

    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr || b->values == nullptr)
    {
        return hcsparseInvalid;
    }

    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;

    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(A, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(A, control);
    }

    hcsparseStatus status = bicgStab<T>(x, A, b, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;
}

hcsparseStatus
hcsparseDcsrbicgStab (hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                      hcsparseSolverControl *solverControl, hcsparseControl *control)
{
    using T = double;

    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr || b->values == nullptr)
    {
        return hcsparseInvalid;
    }

    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;

    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(A, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(A, control);
    }

    hcsparseStatus status = bicgStab<T>(x, A, b, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;
}

hcsparseStatus
hcsparseScsrcg (hcdenseVector *x,
                const hcsparseCsrMatrix *A,
                const hcdenseVector *b,
                hcsparseSolverControl *solverControl,
                hcsparseControl *control)
{
    using T = float;

    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr || b->values == nullptr)
    {
        return hcsparseInvalid;
    }

    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;

    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(A, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(A, control);
    }

    hcsparseStatus status = cg<T>(x, A, b, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;
}

hcsparseStatus
hcsparseDcsrcg (hcdenseVector *x,
                const hcsparseCsrMatrix *A,
                const hcdenseVector *b,
                hcsparseSolverControl *solverControl,
                hcsparseControl *control)
{
    using T = double;

    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (x->values == nullptr || b->values == nullptr)
    {
        return hcsparseInvalid;
    }

    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;

    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(A, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(A, control);
    }

    hcsparseStatus status = cg<T>(x, A, b, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;
}

hcsparseStatus
hcsparseScoo2csr (const hcsparseCooMatrix* coo,
                  hcsparseCsrMatrix* csr,
                  hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (coo->values == nullptr || csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return coo2csr<float> (coo, csr, control);
}

hcsparseStatus
hcsparseDcoo2csr (const hcsparseCooMatrix* coo,
                  hcsparseCsrMatrix* csr,
                  hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (coo->values == nullptr || csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return coo2csr<double> (coo, csr, control);
}

hcsparseStatus
hcsparseScsr2coo (const hcsparseCsrMatrix* csr,
                  hcsparseCooMatrix* coo,
                  hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr || coo->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csr2coo<float> (csr, coo, control);
}

hcsparseStatus
hcsparseDcsr2coo (const hcsparseCsrMatrix* csr,
                  hcsparseCooMatrix* coo,
                  hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr || coo->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csr2coo<double> (csr, coo, control);
}

hcsparseStatus_t hcsparseScsr2dense(hcsparseHandle_t handle,
                                    int m, 
                                    int n, 
                                    const hcsparseMatDescr_t descrA,  
                                    const float *csrValA, 
                                    const int *csrRowPtrA, 
                                    const int *csrColIndA,
                                    float *A, 
                                    int lda)
{
  if (handle == nullptr) 
    return HCSPARSE_STATUS_NOT_INITIALIZED;

  if (!csrValA || !csrRowPtrA || !csrColIndA || !A)
    return HCSPARSE_STATUS_ALLOC_FAILED;

  if (descrA.MatrixType != HCSPARSE_MATRIX_TYPE_GENERAL)
    return HCSPARSE_STATUS_INVALID_VALUE;

  ulong dense_size = m * n;

  const int *offsets = static_cast<const int*>(csrRowPtrA);
  const int *indices = static_cast<const int*>(csrColIndA);
  const float *values = static_cast<const float*>(csrValA);

  float *Avalues = static_cast<float*>(A);

  // temp code 
  // TODO : Remove this in the future
  hcsparseControl control(handle->currentAcclView);
  hcsparseStatus stat = hcsparseSuccess;

  fill_zero<float>(dense_size, Avalues, &control);

  stat = transform_csr_2_dense<float> (dense_size, offsets, indices, values,
                                   m, n, Avalues, &control);
  if (stat != hcsparseSuccess)
    return HCSPARSE_STATUS_EXECUTION_FAILED;

  return HCSPARSE_STATUS_SUCCESS;
}
hcsparseStatus
hcsparseScsr2dense (const hcsparseCsrMatrix* csr,
                    hcdenseMatrix* A,
                    hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csr2dense<float> (csr, A, control);
}

hcsparseStatus
hcsparseDcsr2dense (const hcsparseCsrMatrix* csr,
                    hcdenseMatrix* A,
                    hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return csr2dense<double> (csr, A, control);
}

hcsparseStatus
hcsparseSdense2csr (const hcdenseMatrix* A,
                    hcsparseCsrMatrix* csr,
                    hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return dense2csr<float> (A, csr, control);
}

hcsparseStatus
hcsparseDdense2csr (const hcdenseMatrix* A,
                    hcsparseCsrMatrix* csr,
                    hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
        return hcsparseInvalid;
    }

    if (csr->values == nullptr)
    {
        return hcsparseInvalid;
    }

    return dense2csr<double> (A, csr, control);
}

hcsparseStatus
hcsparseScsrSpGemm (const hcsparseCsrMatrix* sparseMatA,
                    const hcsparseCsrMatrix* sparseMatB,
                    hcsparseCsrMatrix* sparseMatC,
                    hcsparseControl* control)
{
    if (!hcsparseInitialized)
    {
       return hcsparseInvalid;
    }

    if (sparseMatA->values == nullptr || sparseMatB->values == nullptr || sparseMatC->values == nullptr)
    {
       return hcsparseInvalid;
    }

    return csrSpGemm<float> (sparseMatA, sparseMatB, sparseMatC, control);
}

