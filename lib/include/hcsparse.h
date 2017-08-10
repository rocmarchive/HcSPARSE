#ifndef _HC_SPARSE_H_
#define _HC_SPARSE_H_

//#include "hcsparse_struct.h"

//2.2.1. hcsparseHandle_t

// The hcsparseHandle_t type is a pointer to an opaque structure holding the hcsparse library context. 
// The hcsparse library context must be initialized using hcsparseCreate() and the returned handle must be 
// passed to all subsequent library function calls. The context should be destroyed at the end using 
// hcsparseDestroy().

typedef struct hcsparseLibrary* hcsparseHandle_t;
typedef int hcsparseStream_t;

// 2.2.2. hcsparseStatus_t

// The type  hcsparseStatus  is used for function status returns. HCSPARSE 
// helper functions return status directly, while the status of HCSPARSE 
// core functions can be retrieved via  hcsparseGetError() . Currently, the 
// following values are defined: 

enum hcsparseStatus_t : uint32_t {
  HCSPARSE_STATUS_SUCCESS,          // Function succeeds
  HCSPARSE_STATUS_NOT_INITIALIZED,  // HCSPARSE library not initialized
  HCSPARSE_STATUS_ALLOC_FAILED,     // resource allocation failed
  HCSPARSE_STATUS_INVALID_VALUE,    // unsupported numerical value was passed to function
  HCSPARSE_STATUS_MAPPING_ERROR,    // access to GPU memory space failed
  HCSPARSE_STATUS_EXECUTION_FAILED, // GPU program failed to execute
  HCSPARSE_STATUS_INTERNAL_ERROR    // an internal HCSPARSE operation failed
};

// 2.2.3. hcsparseDiagType_t

// This type indicates if the matrix diagonal entries are unity. 
// The diagonal elements are always assumed to be present, but if HCSPARSE_DIAG_TYPE_UNIT 
// is passed to an API routine, then the routine assumes that all diagonal entries are unity
// and will not read or modify those entries. 
// Note that in this case the routine assumes the diagonal entries are equal to one, 
// regardless of what those entries are actually set to in memory.

enum hcsparseDiagType_t :uint32_t {
  HCSPARSE_DIAG_TYPE_NON_UNIT, // the matrix diagonal has non-unit elements.
  HCSPARSE_DIAG_TYPE_UNIT      // the matrix diagonal has unit elements.
};

// 2.2.4. hcsparseFillMode_t

// This type indicates if the lower or upper part of a matrix is stored in sparse storage.

enum hcsparseFillMode_t : uint32_t{
  HCSPARSE_FILL_MODE_LOWER,  // the lower triangular part is stored.
  HCSPARSE_FILL_MODE_UPPER   // the upper triangular part is stored.
};

// 2.2.5 hcsparseIndexBase_t

// This type indicates if the base of the matrix indices is zero or one.

enum hcsparseIndexBase_t : uint32_t {
  HCSPARSE_INDEX_BASE_ZERO,  // the base index is zero.
  HCSPARSE_INDEX_BASE_ONE
};

// 2.2.6. hcsparseMatrixType_t

// This type indicates the type of matrix stored in sparse storage. 
// Notice that for symmetric, Hermitian and triangular matrices only their lower 
// or upper part is assumed to be stored.

enum hcsparseMatrixType_t:uint32_t{
  HCSPARSE_MATRIX_TYPE_GENERAL,    // the matrix is general.
  HCSPARSE_MATRIX_TYPE_SYMMETRIC,  // the matrix is symmetric.
  HCSPARSE_MATRIX_TYPE_HERMITIAN,  // the matrix is hermitian.
  HCSPARSE_MATRIX_TYPE_TRIANGULAR  // the matrix is triangular.
};

// 2.2.7. hcsparseMatDescr_t

// This structure is used to describe the shape and properties of a matrix.

struct hcsparseMatDescr{
    hcsparseMatrixType_t MatrixType;
    hcsparseFillMode_t FillMode;
    hcsparseDiagType_t DiagType;
    hcsparseIndexBase_t IndexBase;
};
typedef struct hcsparseMatDescr* hcsparseMatDescr_t;

// 2.2.8. hcsparseOperation_t

// This type indicates which operations need to be performed with the sparse matrix.

enum hcsparseOperation_t : uint32_t {
  HCSPARSE_OPERATION_NON_TRANSPOSE,
  HCSPARSE_OPERATION_TRANSPOSE,
  HCSPARSE_OPERATION_CONJUGATE_TRANSPOSE
};

// 2.2.9 hcsparseDirection_t

// This type indicates whether the elements of a dense matrix
// should be parsed by rows or by columns in function hcsparse[S|D|C|Z]nnz.

enum hcsparseDirection_t : uint32_t{
  HCSPARSE_DIRECTION_ROW,
  HCSPARSE_DIRECTION_COLUMN
};

// hcsparse Helper functions 

// 1. hcsparseCreate()

// This function initializes the HCSPARSE library and creates a handle to an opaque structure
// holding the HCSPARSE library context.
// Create the handle for use on the specified GPU.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_ALLOC_FAILED       the resources could not be allocated  

hcsparseStatus_t hcsparseCreate(hcsparseHandle_t* handle, hc::accelerator_view *accl_view);

// 2. hcsparseDestory()

// This function releases hardware resources used by the HCSPARSE library.
// This function is usually the last call with a particular handle to the HCSPARSE library.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            the shut down succeeded
// HCSPARSE_STATUS_NOT_INITIALIZED    the library was not initialized

hcsparseStatus_t hcsparseDestroy(hcsparseHandle_t* handle);

// 3. hcsparseSetAcclView()

//This function sets the hcSPARSE library stream, which will be used to execute all subsequent calls to the hcSPARSE library functions. If the hcSPARSE library stream is not set, all kernels use the defaultNULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the hcSPARSE library stream back to NULL.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS         :the stream was set successfully
// HCSPARSE_STATUS_NOT_INITIALIZED :the library was not initialized
hcsparseStatus_t hcsparseSetAcclView(hcsparseHandle_t handle, hc::accelerator_view accl_view, void* stream);

 // 4. hcsparseGetAcclView()

// This function gets the hcSPARSE library stream, which is being used to execute all calls to the hcSPARSE library functions. If the hcSPARSE library stream is not set, all kernels use the defaultNULL stream.

// Return Values
// ---------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS : the stream was returned successfully
// HCSPARSE_STATUS_NOT_INITIALIZED : the library was not initialized

hcsparseStatus_t  hcsparseGetAcclView(hcsparseHandle_t handle, hc::accelerator_view **accl_view, void** stream);
 

// 5. hcsparseCreateMatDescr()

// This function initializes the matrix descriptor. It sets the fields MatrixType
// and IndexBase to the default values HCSPARSE_MATRIX_TYPE_GENERAL and
// HCSPARSE_INDEX_BASE_ZERO, respectively, while leaving other fields uninitialized.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_ALLOC_FAILED       the resources could not be allocated  

hcsparseStatus_t
hcsparseCreateMatDescr(hcsparseMatDescr_t *descrA);

// 6. hcsparseDestroyMatDescr()

// This function releases the memory allocated for the matrix descriptor.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded

hcsparseStatus_t
hcsparseDestroyMatDescr(hcsparseMatDescr_t descrA);

// 7. hcsparseSetMatType()

// This function sets the MatrixType field of the matrix descriptor descrA.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_INVALID_VALUE	An invalid type parameter was passed.

hcsparseStatus_t
hcsparseSetMatType(hcsparseMatDescr_t descrA, hcsparseMatrixType_t type);

// 8. hcsparseSetMatIndexBase()

// This function sets the IndexBase field of the matrix descriptor descrA.

// Return Values
// --------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS            initialization succeeded
// HCSPARSE_STATUS_INVALID_VALUE	An invalid type parameter was passed.

hcsparseStatus_t
hcsparseSetMatIndexBase(hcsparseMatDescr_t descrA, 
                        hcsparseIndexBase_t base);

// 7. hcsparseXcsrmm()

// This function performs one of the following matrix-matrix operations:
// C = α ∗ op ( A ) ∗ B + β ∗ C
    
// A is an m×k sparse matrix that is defined in CSR storage format 
// by the three arrays csrValA, csrRowPtrA, and csrColIndA); 
// B and C are dense matrices; α  and  β are scalars; and

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseScsrmm(hcsparseHandle_t handle, 
               hcsparseOperation_t transA,
               int m, int n, int k, int nnz, 
               const float *alpha, 
               const hcsparseMatDescr_t descrA, 
               const float *csrValA, const int *csrRowPtrA, 
               const int *csrColIndA, const float *B, 
               int ldb, const float *beta, float *C, int ldc);

hcsparseStatus_t
hcsparseDcsrmm(hcsparseHandle_t handle,
               hcsparseOperation_t transA,
               int m, int n, int k, int nnz,
               const double *alpha,
               const hcsparseMatDescr_t descrA,
               const double *csrValA, const int *csrRowPtrA,
               const int *csrColIndA, const double *B,
               int ldb, const double *beta, double *C, int ldc);


// 8. hcsparseXcsr2dense()

// This function converts the sparse matrix in CSR format 
// (that is defined by the 3 arrays csrValA, csrRowPtrA, and csrColIndA)
// into the matrix A in dense format. The dense matrix A is filled
// in with the values of the sparse matrix and with zeros elsewhere.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t
hcsparseScsr2dense(hcsparseHandle_t handle,
                   int m,
                   int n,
                   const hcsparseMatDescr_t descrA,
                   const float *csrValA,
                   const int *csrRowPtrA,
                   const int *csrColIndA,
                   float *A,
                   int lda);

hcsparseStatus_t
hcsparseDcsr2dense(hcsparseHandle_t handle,
                   int m, 
                   int n, 
                   const hcsparseMatDescr_t descrA,  
                   const double *csrValA, 
                   const int *csrRowPtrA, 
                   const int *csrColIndA,
                   double *A, 
                   int lda);

// 9. hcsparseXdense2csr()

// This function converts the matrix A in dense format into a sparse matrix in CSR format.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseSdense2csr(hcsparseHandle_t handle,
                   int m,
                   int n, 
                   const hcsparseMatDescr_t descrA, 
                   const float *A, 
                   int lda,
                   const int *nnzPerRow, 
                   float *csrValA, 
                   int *csrRowPtrA,
                   int *csrColIndA);

hcsparseStatus_t 
hcsparseDdense2csr(hcsparseHandle_t handle,
                   int m,
                   int n, 
                   const hcsparseMatDescr_t descrA, 
                   const double *A, 
                   int lda,
                   const int *nnzPerRow, 
                   double *csrValA, 
                   int *csrRowPtrA,
                   int *csrColIndA);

// 10. hcsparseXcsrgemm()

// This function performs following matrix-matrix operation:
// C = op ( A ) ∗ op ( B )
// where A, B and C are m×k, k×n, and m×n sparse matrices 

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

// TODO: nnz is unused, as it is calculated in the existing API

hcsparseStatus_t
hcsparseScsrgemm(hcsparseHandle_t handle,
                 hcsparseOperation_t transA,
                 hcsparseOperation_t transB,
                 int m,
                 int n,
                 int k,
                 const hcsparseMatDescr_t descrA,
                 const int nnzA,
                 const float *csrValA,
                 const int *csrRowPtrA,
                 const int *csrColIndA,
                 const hcsparseMatDescr_t descrB,
                 const int nnzB,
                 const float *csrValB,
                 const int *csrRowPtrB,
                 const int *csrColIndB,
                 const hcsparseMatDescr_t descrC,
                 float *csrValC,
                 const int *csrRowPtrC,
                 int *csrColIndC);

hcsparseStatus_t
hcsparseDcsrgemm(hcsparseHandle_t handle,
                 hcsparseOperation_t transA,
                 hcsparseOperation_t transB,
                 int m,
                 int n,
                 int k,
                 const hcsparseMatDescr_t descrA,
                 const int nnzA,
                 const double *csrValA,
                 const int *csrRowPtrA,
                 const int *csrColIndA,
                 const hcsparseMatDescr_t descrB,
                 const int nnzB,
                 const double *csrValB,
                 const int *csrRowPtrB,
                 const int *csrColIndB,
                 const hcsparseMatDescr_t descrC,
                 double *csrValC,
                 const int *csrRowPtrC,
                 int *csrColIndC);


// 11. hcsparseXnnz()

// This function computes the number of nonzero elements per
// row or column and the total number of nonzero elements in a dense matrix.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseSnnz(hcsparseHandle_t handle,
             hcsparseDirection_t dirA,
             int m, 
             int n,
             const hcsparseMatDescr_t descrA, 
             const float *A, 
             int lda,
             int *nnzPerRowColumn,
             int *nnzTotalDevHostPtr);

hcsparseStatus_t 
hcsparseDnnz(hcsparseHandle_t handle,
             hcsparseDirection_t dirA,
             int m, 
             int n,
             const hcsparseMatDescr_t descrA, 
             const double *A, 
             int lda,
             int *nnzPerRowColumn,
             int *nnzTotalDevHostPtr);

// 12. hcsparseSdot()

// This function returns the dot product of a vector x in sparse format
// and vector y in dense format. This operation can be written as
//           result = y T x

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseSdoti(hcsparseHandle_t handle, int nnz, 
              const float *xVal, 
              const int *xInd, const float *y, 
              float *resultDevHostPtr, 
              hcsparseIndexBase_t idxBase);

hcsparseStatus_t 
hcsparseDdoti(hcsparseHandle_t handle, int nnz, 
              const double *xVal, 
              const int *xInd, const double *y, 
              double *resultDevHostPtr, 
              hcsparseIndexBase_t idxBase);

// 13. hcsparseXcsc2dense()

// This function converts the sparse matrix in CSC format that is defined
// by the three arrays cscValA, cscColPtrA, and cscRowIndA into the matrix
// A in dense format. The dense matrix A is filled in with the values
// of the sparse matrix and with zeros elsewhere.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseScsc2dense(hcsparseHandle_t handle, int m, int n, 
                   const hcsparseMatDescr_t descrA, 
                   const float *cscValA, 
                   const int *cscRowIndA, const int *cscColPtrA,
                   float *A, int lda);

hcsparseStatus_t 
hcsparseDcsc2dense(hcsparseHandle_t handle, int m, int n, 
                   const hcsparseMatDescr_t descrA, 
                   const double *cscValA, 
                   const int *cscRowIndA, const int *cscColPtrA,
                   double *A, int lda);

// 14. hcsparseXdense2csc

// This function converts the sparse matrix in CSC format 
// that is defined by the three arrays cscValA, cscColPtrA, and cscRowIndA 
// into the matrix A in dense format. The dense matrix A is filled
// in with the values of the sparse matrix and with zeros elsewhere.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED         the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseSdense2csc(hcsparseHandle_t handle, int m, int n, 
                   const hcsparseMatDescr_t descrA, 
                   const float           *A, 
                   int lda, const int *nnzPerCol, 
                   float           *cscValA, 
                   int *cscRowIndA, int *cscColPtrA);

hcsparseStatus_t 
hcsparseDdense2csc(hcsparseHandle_t handle, int m, int n, 
                   const hcsparseMatDescr_t descrA, 
                   const double *A, 
                   int lda, const int *nnzPerCol, 
                   double    *cscValA, 
                   int *cscRowIndA, int *cscColPtrA);

// 15. hcsparseXcsr2coo()

// This function converts the array containing the compressed row 
// pointers (corresponding to CSR format) into an array of
//  uncompressed row indices (corresponding to COO format).

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseXcsr2coo(hcsparseHandle_t handle, const int *csrRowPtr,
                 int nnz, int m, int *cooRowInd,
                 hcsparseIndexBase_t idxBase);

// 16. hcsparseXcoo2csr()

// This function converts the array containing the uncompressed
// row indices (corresponding to COO format) into an array of
// compressed row pointers (corresponding to CSR format).

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED      the library was not initialized.
// HCSPARSE_STATUS_INVALID_VALUE        invalid parameters were passed (m, n, k, nnz<0 or ldb and ldc are incorrect).
// HCSPARSE_STATUS_EXECUTION_FAILED     the function failed to launch on the GPU.

hcsparseStatus_t 
hcsparseXcoo2csr(hcsparseHandle_t handle, const int *cooRowInd,
                 int nnz, int m, int *csrRowPtr, hcsparseIndexBase_t idxBase);


// 17. hcsparseXcsrmv()

// This function performs the matrix-vector operation
// y = α ∗ op ( A ) ∗ x + β ∗ y

// A is an m×n sparse matrix that is defined in CSR storage
// format by the three arrays csrValA, csrRowPtrA, and csrColIndA;
// x and y are vectors; α  and  β are scalars;

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED 	the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED 	the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE 	invalid parameters were passed (m,n,nnz<0).
// HCSPARSE_STATUS_ARCH_MISMATCH 	the device does not support double precision.
// HCSPARSE_STATUS_INTERNAL_ERROR 	an internal operation failed.
// HCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED 	the matrix type is not supported.

hcsparseStatus_t 
hcsparseScsrmv(hcsparseHandle_t handle, hcsparseOperation_t transA, 
               int m, int n, int nnz, const float           *alpha, 
               const hcsparseMatDescr_t descrA, 
               const float           *csrValA, 
               const int *csrRowPtrA, const int *csrColIndA,
               const float           *x, const float           *beta, 
               float           *y);

hcsparseStatus_t 
hcsparseDcsrmv(hcsparseHandle_t handle, hcsparseOperation_t transA, 
               int m, int n, int nnz, const double          *alpha, 
               const hcsparseMatDescr_t descrA, 
               const double          *csrValA, 
               const int *csrRowPtrA, const int *csrColIndA,
               const double          *x, const double          *beta, 
               double          *y);


// 18. hcsparseXcsrgeam()

// This function performs following matrix-matrix operation
// C = α ∗ A + β ∗ B

// where A, B, and C are m×n sparse matrices
// (defined in CSR storage format by the three arrays
//  csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC,
// and csrColIndA|csrColIndB|csrcolIndC respectively), and α and β are scalars.

// Return Values
// ----------------------------------------------------------------------
// HCSPARSE_STATUS_SUCCESS              the operation completed successfully.
// HCSPARSE_STATUS_NOT_INITIALIZED 	the library was not initialized.
// HCSPARSE_STATUS_ALLOC_FAILED 	the resources could not be allocated.
// HCSPARSE_STATUS_INVALID_VALUE 	invalid parameters were passed (m,n,nnz<0).
// HCSPARSE_STATUS_ARCH_MISMATCH 	the device does not support double precision.
// HCSPARSE_STATUS_INTERNAL_ERROR 	an internal operation failed.
// HCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED 	the matrix type is not supported.

hcsparseStatus_t
hcsparseScsrgeam(hcsparseHandle_t handle, 
                 int m, 
                 int n,
                 const float *alpha,
                 const hcsparseMatDescr_t descrA, 
                 int nnzA,
                 const float *csrValA, 
                 const int *csrRowPtrA, 
                 const int *csrColIndA,
                 const float *beta,
                 const hcsparseMatDescr_t descrB, 
                 int nnzB,
                 const float *csrValB, 
                 const int *csrRowPtrB, 
                 const int *csrColIndB,
                 const hcsparseMatDescr_t descrC,
                 float *csrValC, 
                 int *csrRowPtrC, 
                 int *csrColIndC);

hcsparseStatus_t
hcsparseDcsrgeam(hcsparseHandle_t handle, 
                 int m, 
                 int n,
                 const double *alpha,
                 const hcsparseMatDescr_t descrA, 
                 int nnzA,
                 const double *csrValA, 
                 const int *csrRowPtrA, 
                 const int *csrColIndA,
                 const double *beta,
                 const hcsparseMatDescr_t descrB, 
                 int nnzB,
                 const double *csrValB, 
                 const int *csrRowPtrB, 
                 const int *csrColIndB,
                 const hcsparseMatDescr_t descrC,
                 double *csrValC, 
                 int *csrRowPtrC, 
                 int *csrColIndC);


    /*!
    * \brief Initialize the hcsparse library
    * \note Must be called before any other hcsparse API function is invoked.
    *
    * \returns hcsparseSuccess
    *
    * \ingroup SETUP
    */
    hcsparseStatus hcsparseSetup( void );

    /*!
    * \brief Finalize the usage of the hcsparse library
    * Frees all state allocated by the hcsparse runtime and other internal data
    *
    * \returns hcsparseSuccess
    *
    * \ingroup SETUP
    */
    hcsparseStatus hcsparseTeardown( void );

    /*!
    * \brief Initialize a scalar structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] scalar  Scalar structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcsparseInitScalar( hcsparseScalar* scalar );

    /*!
    * \brief Initialize a dense vector structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] vec  Dense vector structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcsparseInitVector( hcdenseVector* vec );

    /*!
    * \brief Initialize a sparse matrix COO structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] cooMatx  Sparse COO matrix structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcsparseInitCooMatrix( hcsparseCooMatrix* cooMatx );

    /*!
    * \brief Initialize a sparse matrix CSR structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] csrMatx  Sparse CSR matrix structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcsparseInitCsrMatrix( hcsparseCsrMatrix* csrMatx );

    /*!
    * \brief Initialize a sparse matrix CSC structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] cscMatx  Sparse CSC matrix structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcsparseInitCscMatrix( hcsparseCscMatrix* cscMatx );

    /*!
    * \brief Initialize a dense matrix structure to be used in the hcsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] denseMatx  Dense matrix structure to be initialized
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup INIT
    */
    hcsparseStatus hcdenseInitMatrix( hcdenseMatrix* denseMatx );
    /**@}*/


    /*!
    * \brief Create a hcsparseSolverControl object to *control hcsparse iterative
    * solver operations
    *
    * \param[in] precond  A valid enumeration constant from PRECONDITIONER
    * \param[in] maxIters  Maximum number of iterations to converge before timing out
    * \param[in] relTol  Relative tolerance
    * \param[in] absTol  Absolute tolerance
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup SOLVER
    */
    hcsparseSolverControl*
        hcsparseCreateSolverControl( PRECONDITIONER precond, int maxIters,
                                     double relTol, double absTol );

     /*!
     * \brief Release a hcsparseSolverControl object created with hcsparseCreateSolverControl
     *
     * \param[in,out] solverControl  hcsparse object created with hcsparseCreateSolverControl
     *
     * \returns \b hcsparseSuccess
     *
     * \ingroup SOLVER
     */
    hcsparseStatus
        hcsparseReleaseSolverControl( hcsparseSolverControl* solverControl );

    /*!
    * \brief Set hcsparseSolverControl state
    *
    * \param[in] solverControl  hcsparse object created with hcsparseCreateSolverControl
    * \param[in] precond A valid enumeration constant from PRECONDITIONER, how to precondition sparse data
    * \param[in] maxIters  Maximum number of iterations to converge before timing out
    * \param[in] relTol  Relative tolerance
    * \param[in] absTol  Absolute tolerance
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup SOLVER
    */
    hcsparseStatus
        hcsparseSetSolverParams( hcsparseSolverControl* solverControl,
                                 PRECONDITIONER precond,
                                 int maxIters, double relTol, double absTol );

     /*!
     * \brief Set the verbosity level of the hcsparseSolverControl object
     *
     * \param[in] solverControl  hcsparse object created with hcsparseCreateSolverControl
     * \param[in] mode A valid enumeration constant from PRINT_MODE, to specify verbosity level
     *
     * \returns \b hcsparseSuccess
     *
     * \ingroup SOLVER
     */
    hcsparseStatus
        hcsparseSolverPrintMode( hcsparseSolverControl* solverControl, PRINT_MODE mode );

    /*!
    * \brief Execute a single precision Conjugate Gradients solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a hcsparse CSR matrix with single precision data
    * \param[in] b  the input dense vector with single precision data
    * \param[in] solverControl  a valid hcsparseSolverControl object created with hcsparseCreateSolverControl
    * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup SOLVER
    */
    hcsparseStatus
        hcsparseScsrcg( hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                        hcsparseSolverControl* solverControl, hcsparseControl *control );

    /*!
    * \brief Execute a double precision Conjugate Gradients solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a hcsparse CSR matrix with double precision data
    * \param[in] b  the input dense vector with double precision data
    * \param[in] solverControl  a valid hcsparseSolverControl object created with hcsparseCreateSolverControl
    * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup SOLVER
    */
    hcsparseStatus
        hcsparseDcsrcg( hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                        hcsparseSolverControl* solverControl, hcsparseControl *control );

     /*!
     * \brief Execute a single precision Bi-Conjugate Gradients Stabilized solver
     *
     * \param[in] x  the dense vector to solve for
     * \param[in] A  the hcsparse CSR matrix with single precision data
     * \param[in] b  the input dense vector with single precision data
     * \param[in] solverControl  a valid hcsparseSolverControl object created with hcsparseCreateSolverControl
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \returns \b hcsparseSuccess
     *
     * \ingroup SOLVER
     */
    hcsparseStatus
        hcsparseScsrbicgStab( hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                              hcsparseSolverControl* solverControl, hcsparseControl *control );

    /*!
    * \brief Execute a double precision Bi-Conjugate Gradients Stabilized solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a hcsparse CSR matrix with double precision data
    * \param[in] b  the input dense vector with double precision data
    * \param[in] solverControl  a valid hcsparseSolverControl object created with hcsparseCreateSolverControl
    * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
    *
    * \returns \b hcsparseSuccess
    *
    * \ingroup SOLVER
    */
    hcsparseStatus
        hcsparseDcsrbicgStab( hcdenseVector* x, const hcsparseCsrMatrix *A, const hcdenseVector *b,
                              hcsparseSolverControl* solverControl, hcsparseControl *control );
    /**@}*/

    /*!
    * \defgroup FILE Support functions provided to read sparse matrices from file
    *
    * \brief Functions to help read the contents of matrix market files from disk
    */
    /**@{*/

    /*!
    * \brief Read the sparse matrix header from file
    *
    * \param[out] nnz  The number of non-zeroes present in the sparse matrix structure
    * \param[out] row  The number of rows in the sparse matrix
    * \param[out] col  The number of columns in the sparse matrix
    * \param[in] filePath  A path in the file-system to the sparse matrix file
    *
    * \note At this time, only matrix market (.MTX) files are supported
    * \warning The value returned in nnz is the maximum possible number of non-zeroes from the sparse
    * matrix on disk (can be used to allocate memory).  The actual number of non-zeroes may be less,
    * depending if explicit zeroes were stored in file.
    * \returns \b hcsparseSuccess
    *
    * \ingroup FILE
    */
    hcsparseStatus
        hcsparseHeaderfromFile( int* nnz, int* row, int* col, const char* filePath );

    /*!
    * \brief Read sparse matrix data from file in single precision COO format
    * \details This function reads the contents of the sparse matrix file into hcsparseCooMatrix data structure.
    * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
    * This function sorts the values read (on host) by row, then column before copying them into
    * device memory
    * \param[out] cooMatx  The COO sparse structure that represents the matrix in device memory
    * \param[in] filePath  A path in the file-system to the sparse matrix file
    * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
    * \param[in] read_explicit_zeroes If the file contains values explicitly declared zero, this *controls
    * whether they are stored in the COO
    *
    * \note The number of non-zeroes actually read from the file may be different than the number of
    * non-zeroes reported from the file header. Symmetrix matrices may store up to twice as many non-zero
    * values compared to the number of values in the file. Explicitly declared zeroes may be stored
    * or not depending on the input \p read_explicit_zeroes.
    * \note The OpenCL device memory must be allocated before the call to this function.
    * \post The sparse data is sorted first by row, then by column.
    * \returns \b hcsparseSuccess
    *
    * \ingroup FILE
    */
    hcsparseStatus
        hcsparseSCooMatrixfromFile( hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes );

    /*!
     * \brief Read sparse matrix data from file in double precision COO format
     * \details This function reads the contents of the sparse matrix file into hcsparseCooMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory.  If the data on disk is stored in single precision, this function will
     * up-convert the values to double.
     * \param[out] cooMatx  The COO sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \param[in] read_explicit_zeroes If the file contains values explicitly declared zero, this *controls
     * whether they are stored in the COO
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header. Symmetrix matrices may store up to twice as many non-zero
     * values compared to the number of values in the file. Explicitly declared zeroes may be stored
     * or not depending on the input \p read_explicit_zeroes.
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \post The sparse data is sorted first by row, then by column.
     * \returns \b hcsparseSuccess
     *
     * \ingroup FILE
     */
    hcsparseStatus
        hcsparseDCooMatrixfromFile( hcsparseCooMatrix* cooMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes );

    /*!
     * \brief Read sparse matrix data from file in single precision CSR format
     * \details This function reads the contents of the sparse matrix file into hcsparseCsrMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory
     * \param[out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \param[in] read_explicit_zeroes If the file contains values explicitly declared zero, this *controls
     * whether they are stored in the CSR
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header. Symmetrix matrices may store up to twice as many non-zero
     * values compared to the number of values in the file. Explicitly declared zeroes may be stored
     * or not depending on the input \p read_explicit_zeroes.
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \post The sparse data is sorted first by row, then by column.
     * \returns \b hcsparseSuccess
     *
     * \ingroup FILE
     */
    hcsparseStatus
        hcsparseSCsrMatrixfromFile( hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes );

    /*!
     * \brief Read sparse matrix data from file in double precision CSR format
     * \details This function reads the contents of the sparse matrix file into hcsparseCsrMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory.  If the data on disk is stored in single precision, this function will
     * up-convert the values to double.
     * \param[out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \param[in] read_explicit_zeroes If the file contains values explicitly declared zero, this *controls
     * whether they are stored in the CSR
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header. Symmetrix matrices may store up to twice as many non-zero
     * values compared to the number of values in the file. Explicitly declared zeroes may be stored
     * or not depending on the input \p read_explicit_zeroes.
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \post The sparse data is sorted first by row, then by column
     * \returns \b hcsparseSuccess
     *
     * \ingroup FILE
     */
    hcsparseStatus
        hcsparseDCsrMatrixfromFile( hcsparseCsrMatrix* csrMatx, const char* filePath, hcsparseControl *control, bool read_explicit_zeroes );

    /*!
     * \brief Calculate the amount of device memory required to hold meta-data for csr-adaptive SpM-dV algorithm
     * \details CSR-adaptive is a high performance sparse matrix times dense vector algorithm.  It requires a pre-processing
     * step to calculate meta-data on the sparse matrix.  This meta-data is stored alongside and carried along
     * with the other matrix data.  This function initializes the rowBlockSize member variable of the csrMatx
     * variable with the appropriate size.  The client program is responsible to allocate device memory in rowBlocks
     * of this size before calling into the library compute routines.
     * \param[in,out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup FILE
    */
    hcsparseStatus
        hcsparseCsrMetaSize( hcsparseCsrMatrix* csrMatx, hcsparseControl *control );

    /*!
     * \brief Calculate the meta-data for csr-adaptive SpM-dV algorithm
     * \details CSR-adaptive is a high performance sparse matrix times dense vector algorithm.  It requires a pre-processing
     * step to calculate meta-data on the sparse matrix.  This meta-data is stored alongside and carried along
     * with the other matrix data.  This function calculates the meta data and stores it into the rowBlocks member of
     * the hcsparseCsrMatrix.
     * \param[in,out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] *control  A valid hcsparseControl created with hcsparseCreateControl
     * \note This function assumes that the memory for rowBlocks has already been allocated by client program
     *
     * \ingroup FILE
     */
    hcsparseStatus
        hcsparseCsrMetaCompute( hcsparseCsrMatrix* csrMatx, hcsparseControl *control );
    /**@}*/

    /*!
     * \brief Single precision scale dense vector by a scalar
     * \details \f$ r \leftarrow \alpha \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSscale( hcdenseVector* r,
                       const hcsparseScalar* alpha,
                       const hcdenseVector* y,
                       hcsparseControl *control );

    /*!
     * \brief Double precision scale dense vector by a scalar
     * \details \f$ r \leftarrow \alpha \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDscale( hcdenseVector* r,
                       const hcsparseScalar* alpha,
                       const hcdenseVector* y,
                       hcsparseControl *control );

    /*!
     * \brief Single precision scale dense vector and add dense vector
     * \details \f$ r \leftarrow \alpha \ast x + y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSaxpy( hcdenseVector* r,
                      const hcsparseScalar* alpha, const hcdenseVector* x,
                      const hcdenseVector* y,
                      hcsparseControl *control );

    /*!
     * \brief Double precision scale dense vector and add dense vector
     * \details \f$ r \leftarrow \alpha \ast x + y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDaxpy( hcdenseVector* r,
                      const hcsparseScalar* alpha, const hcdenseVector* x,
                      const hcdenseVector* y,
                      hcsparseControl *control );

    /*!
     * \brief Single precision scale dense vector and add scaled dense vector
     * \details \f$ r \leftarrow \alpha \ast x + \beta \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value for x
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value for y
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSaxpby( hcdenseVector* r,
                       const hcsparseScalar* alpha, const hcdenseVector* x,
                       const hcsparseScalar* beta,
                       const hcdenseVector* y,
                       hcsparseControl *control );

    /*!
     * \brief Double precision scale dense vector and add scaled dense vector
     * \details \f$ r \leftarrow \alpha \ast x + \beta \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value for x
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value for y
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDaxpby( hcdenseVector* r,
                       const hcsparseScalar* alpha, const hcdenseVector* x,
                       const hcsparseScalar* beta,
                       const hcdenseVector* y,
                       hcsparseControl *control );

    /*!
     * \brief Reduce integer elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseIreduce( hcsparseScalar* s,
                        const hcdenseVector* x,
                        hcsparseControl *control );

    /*!
     * \brief Reduce single precision elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSreduce( hcsparseScalar* s,
                        const hcdenseVector* x,
                        hcsparseControl *control );

    /*!
     * \brief Reduce double precision elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDreduce( hcsparseScalar* s,
                        const hcdenseVector* x,
                        hcsparseControl *control );

    /*!
     * \brief Calculate the single precision L1 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSnrm1( hcsparseScalar* s,
                      const hcdenseVector* x,
                      hcsparseControl *control );

    /*!
     * \brief Calculate the double precision L1 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDnrm1( hcsparseScalar *s,
                      const hcdenseVector* x,
                      hcsparseControl *control );

    /*!
     * \brief Calculate the single precision L2 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSnrm2( hcsparseScalar* s,
                      const hcdenseVector* x,
                      hcsparseControl *control );

    /*!
     * \brief Calculate the double precision L2 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDnrm2( hcsparseScalar* s,
                      const hcdenseVector* x,
                      hcsparseControl *control );

    /*!
     * \brief Calculates the single precision dot-product of a dense vector
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSdot( hcsparseScalar* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Calculates the double precision dot-product of a dense vector
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDdot( hcsparseScalar* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

                 /* element-wise operations for dense vectors +, -, *, / */

    /*!
     * \brief Element-wise single precision addition of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSadd( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise double precision addition of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDadd( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise single precision subtraction of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSsub( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise double precision subtraction of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDsub( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise single precision multiplication of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSmul( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise double precision multiplication of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDmul( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise single precision division of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseSdiv( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );

    /*!
     * \brief Element-wise double precision division of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    hcsparseStatus
        hcdenseDdiv( hcdenseVector* r,
                     const hcdenseVector* x,
                     const hcdenseVector* y,
                     hcsparseControl *control );
    /**@}*/

    /*!
     * \brief Single precision CSR sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * If the CSR sparse matrix structure has rowBlocks information included,
     * then the csr-adaptive algorithm is used.  Otherwise, the csr-vector
     * algorithm is used.
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    hcsparseStatus
        hcsparseScsrmv( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        hcsparseControl *control );

    /*!
     * \brief Double precision CSR sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * If the CSR sparse matrix structure has rowBlocks information included,
     * then the csr-adaptive algorithm is used.  Otherwise, the csr-vector
     * algorithm is used.
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    hcsparseStatus
        hcsparseDcsrmv( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        hcsparseControl *control );


    /*!
     * \brief Single precision COO sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input COO sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    hcsparseStatus
        hcsparseScoomv( const hcsparseScalar* alpha,
                        const hcsparseCooMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        hcsparseControl *control );

    /*!
     * \brief Double precision COO sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input COO sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    hcsparseStatus
        hcsparseDcoomv( const hcsparseScalar* alpha,
                        const hcsparseCooMatrix* matx,
                        const hcdenseVector* x,
                        const hcsparseScalar* beta,
                        hcdenseVector* y,
                        hcsparseControl *control );
    /**@}*/

    /*!
     * \defgroup BLAS-3 Sparse L3 BLAS operations
     *
     * \brief Sparse BLAS level 3 routines for sparse matrix dense matrix
     * \details Level 3 BLAS operations are defined by order \f$ N^3 \f$ operations,
     * usually in the form of a matrix times a matrix.
     * \ingroup BLAS
     */
    /**@{*/

    /*!
     * \brief Single precision CSR sparse matrix times dense matrix
     * \details \f$ C \leftarrow \alpha \ast A \ast B + \beta \ast C \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] sparseMatA  Input CSR sparse matrix
     * \param[in] denseMatB  Input dense matrix
     * \param[in] beta  Scalar value to multiply against dense matrix
     * \param[out] denseMatC  Output dense matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \note This routine is currently implemented as a batched level 2 matrix
     * times a vector.
     *
     * \ingroup BLAS-3
    */
    hcsparseStatus
        hcsparseScsrmm( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* sparseMatA,
                        const hcdenseMatrix* denseMatB,
                        const hcsparseScalar* beta,
                        hcdenseMatrix* denseMatC,
                        hcsparseControl *control );

    /*!
     * \brief Double precision CSR sparse matrix times dense matrix
     * \details \f$ C \leftarrow \alpha \ast A \ast B + \beta \ast C \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] sparseMatA  Input CSR sparse matrix
     * \param[in] denseMatB  Input dense matrix
     * \param[in] beta  Scalar value to multiply against dense matrix
     * \param[out] denseMatC  Output dense matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \note This routine is currently implemented as a batched level 2 matrix
     * times a vector.
     *
     * \ingroup BLAS-3
    */
    hcsparseStatus
        hcsparseDcsrmm( const hcsparseScalar* alpha,
                        const hcsparseCsrMatrix* sparseMatA,
                        const hcdenseMatrix* denseMatB,
                        const hcsparseScalar* beta,
                        hcdenseMatrix* denseMatC,
                        hcsparseControl *control );

    /*!
     * \brief Single Precision CSR Sparse Matrix times Sparse Matrix
     * \details \f$ C \leftarrow A \ast B \f$
     * \param[in] sparseMatA Input CSR sparse matrix
     * \param[in] sparseMatB Input CSR sparse matrix
     * \param[out] sparseMatC Output CSR sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The input sparse matrices data must first be sorted by rows, then by columns
     * \ingroup BLAS-3
     */
   hcsparseStatus
      hcsparseScsrSpGemm( const hcsparseCsrMatrix* sparseMatA,
                          const hcsparseCsrMatrix* sparseMatB,
                                hcsparseCsrMatrix* sparseMatC,
                          hcsparseControl *control );
    /**@}*/

    /*!
     * \defgroup CONVERT Matrix conversion routines
     *
     * \brief Sparse matrix routines to convert from one sparse format into another
     * \note Input sparse matrices have to be sorted by row and then by column.
     * The sparse conversion routines provided by hcsparse require this as a pre-condition.  The hcsparse
     * matrix file reading routines `hcsparse_C__MatrixfromFile` guarantee this property as a post-condition.
     */
    /**@{*/

    /*!
     * \brief Convert a single precision CSR encoded sparse matrix into a COO encoded sparse matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] coo  Output COO encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseScsr2coo( const hcsparseCsrMatrix* csr,
                          hcsparseCooMatrix* coo,
                          hcsparseControl *control );

    /*!
     * \brief Convert a double precision CSR encoded sparse matrix into a COO encoded sparse matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] coo  Output COO encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseDcsr2coo( const hcsparseCsrMatrix* csr,
                          hcsparseCooMatrix* coo,
                          hcsparseControl *control );

    /*!
     * \brief Convert a single precision COO encoded sparse matrix into a CSR encoded sparse matrix
     * \param[in] coo  Input COO encoded sparse matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseScoo2csr( const hcsparseCooMatrix* coo,
                          hcsparseCsrMatrix* csr,
                          hcsparseControl *control );

    /*!
     * \brief Convert a double precision COO encoded sparse matrix into a CSR encoded sparse matrix
     * \param[in] coo  Input COO encoded sparse matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseDcoo2csr( const hcsparseCooMatrix* coo,
                          hcsparseCsrMatrix* csr,
                          hcsparseControl *control );

    /*!
     * \brief Convert a single precision CSR encoded sparse matrix into a dense matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] A  Output dense matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseScsr2dense( const hcsparseCsrMatrix* csr,
                            hcdenseMatrix* A,
                            hcsparseControl *control );

    /*!
     * \brief Convert a double precision CSR encoded sparse matrix into a dense matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] A  Output dense matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseDcsr2dense( const hcsparseCsrMatrix* csr,
                            hcdenseMatrix* A,
                            hcsparseControl *control );

    /*!
     * \brief Convert a single precision dense matrix into a CSR encoded sparse matrix
     * \param[in] A  Input dense matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseSdense2csr( const hcdenseMatrix* A,
                            hcsparseCsrMatrix* csr,
                            hcsparseControl *control );

    /*!
     * \brief Convert a double precision dense matrix into a CSR encoded sparse matrix
     * \param[in] A  Input dense matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] *control A valid hcsparseControl created with hcsparseCreateControl
     * \pre The sparse matrix data must first be sorted by rows, then by columns
     *
     * \ingroup CONVERT
     */
    hcsparseStatus
        hcsparseDdense2csr( const hcdenseMatrix* A, hcsparseCsrMatrix* csr,
                            hcsparseControl *control );
    /**@}*/

#endif // _HC_SPARSE_H_
