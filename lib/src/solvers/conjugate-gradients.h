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
cg(hcdenseVector *pX,
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

    Concurrency::array_view<T> *x = static_cast<Concurrency::array_view<T> *>(pX->values);
    Concurrency::array_view<T> *b = static_cast<Concurrency::array_view<T> *>(pB->values);

    int status;

    T *norm_b_Buff = (T*) calloc(1, sizeof(T));
    Concurrency::array_view<T> av_norm_b(1, norm_b_Buff);

    hcsparseScalar norm_b;
    norm_b.value = &av_norm_b;
    norm_b.offValue = 0;

    //norm of rhs of equation
    status = Norm1<T>(&norm_b, pB, control);

    //norm_b is calculated once
    T h_norm_b = av_norm_b[0];

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
            (*x)[i] = (*b)[i];

        return hcsparseSuccess;
    }


    //continuing "normal" execution of cg algorithm
    const auto N = pA->num_cols;

    //helper containers, all need to be zeroed
    T *y_Buff = (T*) calloc(N, sizeof(T));
    T *z_Buff = (T*) calloc(N, sizeof(T));
    T *r_Buff = (T*) calloc(N, sizeof(T));
    T *p_Buff = (T*) calloc(N, sizeof(T));

    Concurrency::array_view<T> av_y(N, y_Buff);
    Concurrency::array_view<T> av_z(N, z_Buff);
    Concurrency::array_view<T> av_r(N, r_Buff);
    Concurrency::array_view<T> av_p(N, p_Buff);

    hcdenseVector y;
    hcdenseVector z;
    hcdenseVector r;
    hcdenseVector p;

    y.values = &av_y;
    z.values = &av_z;
    r.values = &av_r;
    p.values = &av_p;
 
    y.num_values = N;
    z.num_values = N;
    r.num_values = N;
    p.num_values = N;

    y.offValues = 0;
    z.offValues = 0;
    r.offValues = 0;
    p.offValues = 0;

    T *one_Buff = (T*) calloc(1, sizeof(T));
    T *zero_Buff = (T*) calloc(1, sizeof(T));

    one_Buff[0] = 1;
 
    Concurrency::array_view<T> av_one(1, one_Buff);
    Concurrency::array_view<T> av_zero(1, zero_Buff);

    hcsparseScalar one;
    hcsparseScalar zero;

    one.value = &av_one;
    zero.value = &av_zero;
  
    one.offValue = 0;
    zero.offValue = 0;

    // y = A*x
    status = csrmv<T>(&one, pA, pX, &zero, &y, control);

    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);

    T *norm_r_Buff = (T*) calloc(1, sizeof(T));
    Concurrency::array_view<T> av_norm_r(1, norm_r_Buff);

    hcsparseScalar norm_r;
    norm_r.value = &av_norm_r;
    norm_r.offValue = 0;

    status = Norm1<T>(&norm_r, &r, control);

    //T residuum = 0;
    T *residuum_Buff = (T*) calloc(1, sizeof(T));
    Concurrency::array_view<T> av_residuum(1, residuum_Buff);

    hcsparseScalar residuum;
    residuum.value = &av_residuum;
    residuum.offValue = 0;

    //residuum = norm_r[0] / h_norm_b;
    av_residuum[0] = div<T>(av_norm_r[0], av_norm_b[0]);

    solverControl->initialResidual = av_residuum[0];
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
    for (int i = 0; i < N; i++)
        av_p[i] = av_z[i];

    //rz = <r, z>, here actually should be conjugate(r)) but we do not support complex type.
    T *rz_Buff = (T*) calloc(1, sizeof(T));
    Concurrency::array_view<T> av_rz(1, rz_Buff);

    hcsparseScalar rz;
    rz.value = &av_rz;
    rz.offValue = 0;

    status = dot<T>(&rz, &r, &z, control);

    int iteration = 0;

    bool converged = false;

    T *alpha_Buff = (T*) calloc(1, sizeof(T));
    T *beta_Buff = (T*) calloc(1, sizeof(T));
    T *yp_Buff = (T*) calloc(1, sizeof(T));
    T *rz_old_Buff = (T*) calloc(1, sizeof(T));

    Concurrency::array_view<T> av_alpha(1, alpha_Buff);
    Concurrency::array_view<T> av_beta(1, beta_Buff);
    Concurrency::array_view<T> av_yp(1, yp_Buff);
    Concurrency::array_view<T> av_rz_old(1, rz_old_Buff);

    hcsparseScalar alpha;
    hcsparseScalar beta;
    hcsparseScalar yp;
    hcsparseScalar rz_old;

    alpha.value = &av_alpha;
    beta.value = &av_beta;
    yp.value = &av_yp;
    rz_old.value = &av_rz_old;

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

        // alpha = <r,z> / <y,p>
        //alpha[0] = rz[0] / yp[0];
        av_alpha[0] = div<T>(av_rz[0], av_yp[0]);

#ifndef NDEBUG
            std::cout << "alpha = " << av_alpha[0] << std::endl;
#endif

        //x = x + alpha*p
        status = axpy<T>(pX, &alpha, &p, pX, control);

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(&r, &alpha, &y, &r, control);

        //apply preconditioner z = M*r
        M(&r, &z, control);

        //store old value of rz
        //improve that by move or swap
        av_rz_old[0] = av_rz[0];

        //rz = <r,z>
        status = dot<T>(&rz, &r, &z, control);

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        //beta[0] = rz[0] / rz_old[0];
        av_beta[0] = div<T>(av_rz[0], av_rz_old[0]);
#ifndef NDEBUG
            std::cout << "beta = " << av_beta[0] << std::endl;
#endif

        //p = z + beta*p;
        status = axpby<T>(&p, &one, &z, &beta, &p, control );

        //calculate norm of r
        status = Norm1<T>(&norm_r, &r, control);

        //residuum = norm_r[0] / h_norm_b;
        av_residuum[0] = div<T>(av_norm_r[0], av_norm_b[0]);

        iteration++;
        converged = solverControl->finished(av_residuum[0]);

        solverControl->print();
    }
    return hcsparseSuccess;
}

#endif //_HCSPARSE_SOLVER_CG_H_
