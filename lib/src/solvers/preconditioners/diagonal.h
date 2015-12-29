#ifndef _HCSPARSE_PREC_DIAGONAL_H_
#define _HCSPARSE_PREC_DIAGONAL_H_

#include "preconditioner.h"
#include "preconditioner_utils.h"

/* The simplest preconditioner consists of just the
   inverse values of the diagonal of the matrix:
   The Jacobi preconditioner is one of the simplest forms of preconditioning,
   in which the preconditioner is chosen to be the diagonal of the matrix
                        P = \mathrm{diag}(A).
    Assuming A_{ii} \neq 0, \forall i ,
    we get P^{-1}_{ij} = \frac{\delta_{ij}}{A_{ij}}.
    It is efficient for diagonally dominant matrices A.
*/

template<typename T>
class DiagonalPreconditioner
{
public:
    DiagonalPreconditioner(const hcsparseCsrMatrix* A,
                           hcsparseControl* control)
    {

        int status;

        int size = std::min(A->num_rows, A->num_cols);

        invBuff = (T*) calloc ( size, sizeof(T));
        Concurrency::array_view<T> av_invBuff(size, invBuff);

        invDiag_A.values = &av_invBuff;
        invDiag_A.num_values = size;
        invDiag_A.offValues = 0;

        // extract inverse diagonal from matrix A and store it in invDiag_A
        // easy to check with poisson matrix;
        status = extract_diagonal<T, true>(&invDiag_A, A, control);
    }

    // apply preconditioner
    void operator ()(const hcdenseVector *x,
                     hcdenseVector *y,
                     hcsparseControl* control)    
    {
        //element wise multiply y = x*invDiag_A;
        hcsparseStatus status =
                elementwise_transform<T, EW_MULTIPLY>(y, x, &invDiag_A, control);
    }

    ~DiagonalPreconditioner()
    {
        free(invBuff);
    }

private:
    //inverse diagonal values of matrix A;
    T *invBuff;
    hcdenseVector invDiag_A;
};


template<typename T>
class DiagonalHandler : public PreconditionerHandler<T>
{
public:

    using Diag = DiagonalPreconditioner<T>;

    DiagonalHandler()
    {
    }

    void operator()(const hcdenseVector *x,
                    hcdenseVector *y,
                    hcsparseControl* control)
    {
        (*diagonal)(x, y, control);
    }

    void notify(const hcsparseCsrMatrix* pA, hcsparseControl* control)
    {
        diagonal = std::make_shared<Diag>(pA, control);
    }

private:
    std::shared_ptr<Diag> diagonal;
};

#endif //_HCSPARSE_PREC_DIAGONAL_H_
