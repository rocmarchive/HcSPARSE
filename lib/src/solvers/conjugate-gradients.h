#ifndef _HCSPARSE_SOLVER_CG_H_
#define _HCSPARSE_SOLVER_CG_H_

#include "hcsparse.h"

/*
 * Nice paper describing Conjugate Gradient algorithm can
 * be found here:
 * http://www.cs.cmu.edu/~./quake-papers/painless-conjugate-gradient.pdf
 */
template<typename T, typename PTYPE>
hcsparseStatus
cg (hcdenseVector *pX,
    const hcsparseCsrMatrix* pA,
    const hcdenseVector *pB,
    PTYPE& M,
    hcsparseSolverControl *solverControl,
    hcsparseControl *control)
{

    assert( pA->num_cols == pB->num_values );
    assert( pA->num_rows == pX->num_values );
    if( ( pA->num_cols != pB->num_values ) || ( pA->num_rows != pX->num_values ) )
    {
        return hcsparseInvalid;
    }

    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *x = static_cast<T*>(pX->values);
    T *b = static_cast<T*>(pB->values);

    int status;

    T *norm_b_Buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar norm_b;
    norm_b.value = am_alloc(sizeof(T)*1, acc, 0);
    norm_b.offValue = 0;

    //norm of rhs of equation
    status = Norm1<T>(&norm_b, pB, control);
    control->accl_view.copy(norm_b.value, norm_b_Buff, sizeof(T)*1);

    //norm_b is calculated once
    T h_norm_b = norm_b_Buff[0];

#ifndef NDEBUG
    std::cout << "norm_b " << h_norm_b << std::endl;
#endif

    if (h_norm_b == 0) //special case b is zero so solution is x = 0
    {
        solverControl->nIters = 0;
        solverControl->absoluteTolerance = 0.0;
        solverControl->relativeTolerance = 0.0;
        //we can either fill the x with zeros or cpy b to x;
        for (int i = 0; i < pX->num_values; i++)
            x[i] = b[i];

        return hcsparseSuccess;
    }


    //continuing "normal" execution of cg algorithm
    const auto N = pA->num_cols;

    //helper containers, all need to be zeroed
    T *y_Buff = (T*) calloc(N, sizeof(T));
    T *z_Buff = (T*) calloc(N, sizeof(T));
    T *r_Buff = (T*) calloc(N, sizeof(T));
    T *p_Buff = (T*) calloc(N, sizeof(T));

    hcdenseVector y;
    hcdenseVector z;
    hcdenseVector r;
    hcdenseVector p;

    y.values = am_alloc(sizeof(T)*N, acc, 0);
    z.values = am_alloc(sizeof(T)*N, acc, 0);
    r.values = am_alloc(sizeof(T)*N, acc, 0);
    p.values = am_alloc(sizeof(T)*N, acc, 0);
 
    y.num_values = N;
    z.num_values = N;
    r.num_values = N;
    p.num_values = N;

    y.offValues = 0;
    z.offValues = 0;
    r.offValues = 0;
    p.offValues = 0;

    T *one_buff = (T*) calloc(1, sizeof(T));

    one_buff[0] = 1;

    hcsparseScalar one;
    hcsparseScalar zero;

    one.value = am_alloc(sizeof(T)*1, acc, 0);
    zero.value = am_alloc(sizeof(T)*1, acc, 0);
    control->accl_view.copy(one_buff, one.value, sizeof(T)*1);

    one.offValue = 0;
    zero.offValue = 0;

    // y = A*x
    status = csrmv<T>(&one, pA, pX, &zero, &y, control);

    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);

    T *norm_r_Buff = (T*) calloc(1, sizeof(T));

    hcsparseScalar norm_r;
    norm_r.value = am_alloc(sizeof(T)*1, acc, 0);
    norm_r.offValue = 0;

    status = Norm1<T>(&norm_r, &r, control);
    control->accl_view.copy(norm_r.value, norm_r_Buff, sizeof(T)*1);

    //T residuum = 0;
    T *residuum_Buff = (T*) calloc(1, sizeof(T));

    //residuum = norm_r[0] / h_norm_b;
    residuum_Buff[0] = div<T>(norm_r_Buff[0], norm_b_Buff[0]);

    solverControl->initialResidual = residuum_Buff[0];
#ifndef NDEBUG
        std::cout << "initial residuum = "
                  << solverControl->initialResidual << std::endl;
#endif
    if (solverControl->finished(solverControl->initialResidual))
    {
        solverControl->nIters = 0;
        return hcsparseSuccess;
    }
    //apply preconditioner z = M*r
    M(&r, &z, control);

    //copy inital z to p
    control->accl_view.copy(z.values, p.values, sizeof(T)*N);

    //rz = <r, z>, here actually should be conjugate(r)) but we do not support complex type.
    T *rz_Buff = (T*) calloc(1, sizeof(T));

    hcsparseScalar rz;
    rz.value = am_alloc(sizeof(T)*1, acc, 0);
    rz.offValue = 0;

    status = dot<T>(&rz, &r, &z, control);
    control->accl_view.copy(rz.value, rz_Buff, sizeof(T)*1);

    int iteration = 0;

    bool converged = false;

    T *alpha_Buff = (T*) calloc(1, sizeof(T));
    T *beta_Buff = (T*) calloc(1, sizeof(T));
    T *yp_Buff = (T*) calloc(1, sizeof(T));
    T *rz_old_Buff = (T*) calloc(1, sizeof(T));

    hcsparseScalar alpha;
    hcsparseScalar beta;
    hcsparseScalar yp;
    hcsparseScalar rz_old;

    alpha.value = am_alloc(sizeof(T)*1, acc, 0);
    beta.value = am_alloc(sizeof(T)*1, acc, 0);
    yp.value = am_alloc(sizeof(T)*1, acc, 0);
    rz_old.value = am_alloc(sizeof(T)*1, acc, 0);

    alpha.offValue = 0;
    beta.offValue = 0;
    yp.offValue = 0;
    rz_old.offValue = 0;

    while(!converged)
    {
        solverControl->nIters = iteration;

        //y = A*p
        status = csrmv<T>(&one, pA, &p, &zero, &y, control);

        status = dot<T>(&yp, &y, &p, control);
        control->accl_view.copy(yp.value, yp_Buff, sizeof(T)*1);

        // alpha = <r,z> / <y,p>
        //alpha[0] = rz[0] / yp[0];
        alpha_Buff[0] = div<T>(rz_Buff[0], yp_Buff[0]);
        control->accl_view.copy(alpha_Buff, alpha.value, sizeof(T)*1);

#ifndef NDEBUG
            std::cout << "alpha = " << alpha_Buff[0] << std::endl;
#endif

        //x = x + alpha*p
        status = axpy<T>(pX, &alpha, &p, pX, control);

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(&r, &alpha, &y, &r, control);

        //apply preconditioner z = M*r
        M(&r, &z, control);

        //store old value of rz
        //improve that by move or swap
        rz_old_Buff[0] = rz_Buff[0];
        control->accl_view.copy(rz.value, rz_old.value, sizeof(T)*1);

        //rz = <r,z>
        status = dot<T>(&rz, &r, &z, control);
        control->accl_view.copy(rz.value, rz_Buff, sizeof(T)*1);

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        //beta[0] = rz[0] / rz_old[0];
        beta_Buff[0] = div<T>(rz_Buff[0], rz_old_Buff[0]);
#ifndef NDEBUG
            std::cout << "beta = " << beta_Buff[0] << std::endl;
#endif

        //p = z + beta*p;
        control->accl_view.copy(beta_Buff, beta.value, sizeof(T)*1);
        status = axpby<T>(&p, &one, &z, &beta, &p, control );

        //calculate norm of r
        status = Norm1<T>(&norm_r, &r, control);
        control->accl_view.copy(norm_r.value, norm_r_Buff, sizeof(T)*1);

        //residuum = norm_r[0] / h_norm_b;
        residuum_Buff[0] = div<T>(norm_r_Buff[0], norm_b_Buff[0]);

        iteration++;
        converged = solverControl->finished(residuum_Buff[0]);

        solverControl->print();
    }
    return hcsparseSuccess;
}

#endif //_HCSPARSE_SOLVER_CG_H_
