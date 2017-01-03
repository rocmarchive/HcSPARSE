#ifndef _HCSPARSE_PREC_VOID_H_
#define _HCSPARSE_PREC_VOID_H_

#include "preconditioner.h"
#include "preconditioner_utils.h"

/*
 * Void preconditioner, does nothing it is just to fit the solver structure.
 * This is the default value of the solver control.
 */
template <typename T>
class VoidPreconditioner
{
public:

    VoidPreconditioner (const hcsparseCsrMatrix* A, hcsparseControl* control)
    {

    }

    void operator() (const hcdenseVector *x,
                     hcdenseVector *y,
                     hcsparseControl* control)
    {

        //assert (x.size() == y.size());

        //void does nothing just copy x to y;

        //deep copy;
        control->accl_view.copy(x->values, y->values, sizeof(T)*x->num_values);
    }

};

template <typename T>
class VoidHandler : public PreconditionerHandler<T>
{
public:

    using Void = VoidPreconditioner<T>;

    VoidHandler()
    {

    }

    void operator ()(const hcdenseVector *x,
                     hcdenseVector *y,
                     hcsparseControl* control)
    {
        (*void_precond)(x, y, control);
    }

    void notify(const hcsparseCsrMatrix *pA,
                hcsparseControl* control)
    {
        void_precond = std::make_shared<Void> (pA, control);
    }

private:
    std::shared_ptr<Void> void_precond;
};

#endif //_HCSPARSE_PREC_VOID_H_
