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
        hc::accelerator acc = (control->accl_view).get_accelerator();
        
        invBuff = (T*) calloc ( size, sizeof(T));
        invDiag_A.values = am_alloc(sizeof(T)*size, acc, 0);
        invDiag_A.num_values = size;
        invDiag_A.offValues = 0;

        // extract inverse diagonal from matrix A and store it in invDiag_A
        // easy to check with poisson matrix;
        status = extract_diagonal<T, true>(&invDiag_A, A, control);

        T *avInvDiag = static_cast<T *>(invDiag_A.values);         
        control->accl_view.copy(invDiag_A.values, invBuff, sizeof(T)*size);
    }

    // apply preconditioner
    void operator ()(const hcdenseVector *x,
                     hcdenseVector *y,
                     hcsparseControl* control)    
    {

        if (invDiag_A.values != NULL)
          control->accl_view.copy(invBuff, invDiag_A.values, sizeof(T)*invDiag_A.num_values);

        //element wise multiply y = x*invDiag_A;
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
