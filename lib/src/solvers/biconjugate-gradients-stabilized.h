#ifndef _HCSPARSE_SOLVER_BICGSTAB_H_
#define _HCSPARSE_SOLVER_BICGSTAB_H_

#include "hcsparse.h"

template <typename T, typename PTYPE>
hcsparseStatus
bicgStab(hcdenseVector *pX,
         const hcsparseCsrMatrix* pA,
         const hcdenseVector *pB,
         PTYPE& M,
         hcsparseSolverControl* solverControl,
         hcsparseControl* control)
{
    assert( pA->num_cols == pB->num_values );
    assert( pA->num_rows == pX->num_values );
    if( ( pA->num_cols != pB->num_values ) || ( pA->num_rows != pX->num_values ) )
    {
        std::cout<<"Size mismatch"<<std::endl;
        return hcsparseInvalid;
    }

    hc::array_view<T> *x = static_cast<hc::array_view<T> *>(pX->values);
    hc::array_view<T> *b = static_cast<hc::array_view<T> *>(pB->values);

    int status;

    T *norm_b_buff = (T*)calloc(1, sizeof(T));
    hc::array_view<T> av_norm_b(1, norm_b_buff);
    hcsparseScalar norm_b;
    norm_b.value = &av_norm_b;
    norm_b.offValue = 0;

    //norm of rhs of equation
    status = Norm1<T>(&norm_b, pB, control);

    //norm_b is calculated once
    T h_norm_b = av_norm_b[0];

    if (h_norm_b <= std::numeric_limits<T>::min())
    {
        solverControl->nIters = 0;
        solverControl->absoluteTolerance = 0.0;
        solverControl->relativeTolerance = 0.0;
        //we can either fill the x with zeros or cpy b to x;
        for (int i = 0; i < pX->num_values; i++)
            (*x)[i] = (*b)[i];

        return hcsparseSuccess;
    }



    //n == number of rows;
    const auto N = pA->num_cols;

    T *y_buff = (T*) calloc(N, sizeof(T));
    T *p_buff = (T*) calloc(N, sizeof(T));
    T *r_buff = (T*) calloc(N, sizeof(T));
    T *r_star_buff = (T*) calloc(N, sizeof(T));
    T *s_buff = (T*) calloc(N, sizeof(T));
    T *Mp_buff = (T*) calloc(N, sizeof(T));
    T *AMp_buff = (T*) calloc(N, sizeof(T));
    T *Ms_buff = (T*) calloc(N, sizeof(T));
    T *AMs_buff = (T*) calloc(N, sizeof(T));

    hc::array_view<T> av_y(N, y_buff);
    hc::array_view<T> av_p(N, p_buff);
    hc::array_view<T> av_r(N, r_buff);
    hc::array_view<T> av_r_star(N, r_star_buff);
    hc::array_view<T> av_s(N, s_buff);
    hc::array_view<T> av_Mp(N, Mp_buff);
    hc::array_view<T> av_AMp(N, AMp_buff);
    hc::array_view<T> av_Ms(N, Ms_buff);
    hc::array_view<T> av_AMs(N, AMs_buff);

    hcdenseVector y;
    hcdenseVector p;
    hcdenseVector r;
    hcdenseVector r_star;
    hcdenseVector s;
    hcdenseVector Mp;
    hcdenseVector AMp;
    hcdenseVector Ms;
    hcdenseVector AMs;

    y.values = &av_y;
    p.values = &av_p;
    r.values = &av_r;
    r_star.values = &av_r_star;
    s.values = &av_s;
    Mp.values = &av_Mp;
    AMp.values = &av_AMp;
    Ms.values = &av_Ms;
    AMs.values = &av_AMs;

    y.num_values = N;
    p.num_values = N;
    r.num_values = N;
    r_star.num_values = N;
    s.num_values = N;
    Mp.num_values = N;
    AMp.num_values = N;
    Ms.num_values = N;
    AMs.num_values = N;

    y.offValues = 0;
    p.offValues = 0;
    r.offValues = 0;
    r_star.offValues = 0;
    s.offValues = 0;
    Mp.offValues = 0;
    AMp.offValues = 0;
    Ms.offValues = 0;
    AMs.offValues = 0;

    T *one_buff = (T*) calloc(1, sizeof(T));
    T *zero_buff = (T*) calloc(1, sizeof(T));

    one_buff[0] = 1;

    hc::array_view<T> av_one(1, one_buff);
    hc::array_view<T> av_zero(1, zero_buff);

    hcsparseScalar one;
    hcsparseScalar zero;

    one.value = &av_one;
    zero.value = &av_zero;

    one.offValue = 0;
    zero.offValue = 0;

    // y = A * x
    status = csrmv<T>(&one, pA, pX, &zero, &y, control);

    // r = b - y
    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);

    T *norm_r_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_norm_r(1, norm_r_buff);

    hcsparseScalar norm_r;

    norm_r.value = &av_norm_r;
    norm_r.offValue = 0;

    status = Norm1<T>(&norm_r, &r, control);

    T *residuum_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_residuum(1, residuum_buff);

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

    // p = r
    for (int i = 0; i < N; i++)
        av_p[i] = av_r[i];

    //Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
    for (int i = 0; i < N; i++)
        av_r_star[i] = av_r[i];

    // holder for <r_star, r>
    T *r_star_r_old_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_r_star_r_old(1, r_star_r_old_buff);

    hcsparseScalar r_star_r_old;
    r_star_r_old.value = &av_r_star_r_old;
    r_star_r_old.offValue = 0;

    // holder for <r_star, r_{i+1}>
    T *r_star_r_new_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_r_star_r_new(1, r_star_r_new_buff);
  
    hcsparseScalar r_star_r_new;
    r_star_r_new.value = &av_r_star_r_new;
    r_star_r_new.offValue = 0;

    status = dot<T>(&r_star_r_old, &r_star, &r, control);

    int iteration = 0;
    bool converged = false;

    T *alpha_buff = (T*) calloc(1, sizeof(T));
    T *beta_buff = (T*) calloc(1, sizeof(T));
    T *omega_buff = (T*) calloc(1, sizeof(T));

    hc::array_view<T> av_alpha(1, alpha_buff);
    hc::array_view<T> av_beta(1, beta_buff);
    hc::array_view<T> av_omega(1, omega_buff);

    hcsparseScalar alpha;
    hcsparseScalar beta;
    hcsparseScalar omega;

    alpha.value = &av_alpha;
    alpha.offValue = 0;
  
    beta.value = &av_beta;
    beta.offValue = 0;
    
    omega.value = &av_omega;
    omega.offValue = 0;

    // holder for <r_star, AMp>
    T *r_star_AMp_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_r_star_AMp(1, r_star_AMp_buff);

    hcsparseScalar r_star_AMp;
    r_star_AMp.value = &av_r_star_AMp;
    r_star_AMp.offValue = 0;

    // hoder for <A*M*s, s>
    T *AMsS_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_AMsS(1, AMsS_buff);

    hcsparseScalar AMsS;
    AMsS.value = &av_AMsS;
    AMsS.offValue = 0;

    // holder for <AMs, AMs>
    T *AMsAMs_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_AMsAMs(1, AMsAMs_buff);

    hcsparseScalar AMsAMs;
    AMsAMs.value = &av_AMsAMs;
    AMsAMs.offValue = 0;

    // holder for norm_s;
    T *norm_s_buff = (T*) calloc(1, sizeof(T));
    hc::array_view<T> av_norm_s(1, norm_s_buff);

    hcsparseScalar norm_s;
    norm_s.value = &av_norm_s;
    norm_s.offValue = 0;

    while (!converged)
    {
        //Mp = M*p //apply preconditioner
        M(&p, &Mp, control);

        //AMp = A*Mp
        status = csrmv<T>(&one, pA, &Mp, &zero, &AMp, control);

        //<r_star, A*M*p>
        status = dot<T>(&r_star_AMp, &r_star, &AMp, control);

        av_alpha[0] = div<T>(av_r_star_r_old[0], av_r_star_AMp[0]);

        //s_j = r - alpha*Mp
        status = axpby<T, EW_MINUS>(&s, &one, &r, &alpha, &AMp, control);

        status = Norm1<T>(&norm_s, &s, control);

        av_residuum[0] = div<T>(av_norm_s[0], av_norm_b[0]);

        if (solverControl->finished(av_residuum[0]))
        {
//            iteration++;
            solverControl->nIters = iteration;
            //x = x + alpha * M*p_j;
            status = axpby<T>(pX, &one, pX, &alpha, &Mp, control);
            break;
        }

        //Ms = M*s
        M(&s, &Ms, control);

        //AMs = A*Ms
        status = csrmv<T>(&one, pA, &Ms, &zero, &AMs, control);

        status = dot<T>(&AMsS, &AMs, &s, control);

        status = dot<T> (&AMsAMs, &AMs, &AMs, control);
        av_omega[0] = div(av_AMsS[0], av_AMsAMs[0]);

#ifndef NDEBUG
        if(av_omega[0] == 0)
            std::cout << "omega = 0" ;
#endif

        //x = x + alpha*Mp + omega*Ms;
        status = axpy<T>(pX, &alpha, &Mp, pX, control);

        status = axpy<T>(pX, &omega, &Ms, pX, control);

        // r = s - omega * A*M*s
        status = axpy<T, EW_MINUS>(&r, &omega, &AMs, &s, control);

        status = Norm1<T>(&norm_r, &r, control);

        av_residuum[0] = div<T>(av_norm_r[0], av_norm_b[0]);

        if (solverControl->finished(av_residuum[0]))
        {
//            iteration++;
            solverControl->nIters = iteration;
            break;
        }

        //beta = <r_star, r+1> / <r_star, r> * (alpha/omega)
        status = dot<T>(&r_star_r_new, &r_star, &r, control);

        //TODO:: is it the best order?
        av_beta[0] = div<T>(av_r_star_r_new[0], av_r_star_r_old[0]);
        av_beta[0] = multi<T>(av_beta[0], av_alpha[0]);
        av_beta[0] = div<T>(av_beta[0], av_omega[0]);

        av_r_star_r_old[0] = av_r_star_r_new[0];

        //p = r + beta* (p - omega A*M*p);
        status = axpy<T>(&p, &beta, &p, &r, control); // p = beta * p + r;
        av_beta[0] = multi<T>(av_beta[0], av_omega[0]);  // (beta*omega)
        status = axpy<T,EW_MINUS>(&p, &beta, &AMp, &p, control);  // p = p - beta*omega*AMp;

        iteration++;
        solverControl->nIters = iteration;

        solverControl->print();
    }

    return hcsparseSuccess;
}

#endif //SOLVER_BICGSTAB_H_
