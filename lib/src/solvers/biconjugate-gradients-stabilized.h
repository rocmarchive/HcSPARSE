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

    hc::accelerator acc = (control->accl_view).get_accelerator();

    T *x = static_cast<T*>(pX->values);
    T *b = static_cast<T*>(pB->values);

    int status;
    T *norm_b_buff = (T*)calloc(1, sizeof(T));
    hcsparseScalar norm_b;
    norm_b.value = am_alloc(sizeof(T)*1, acc, 0);
    norm_b.offValue = 0;

    //norm of rhs of equation
    status = Norm1<T>(&norm_b, pB, control);
    control->accl_view.copy(norm_b.value, norm_b_buff, sizeof(T)*1);

    //norm_b is calculated once
    T h_norm_b = norm_b_buff[0];

    if (h_norm_b <= std::numeric_limits<T>::min())
    {
        solverControl->nIters = 0;
        solverControl->absoluteTolerance = 0.0;
        solverControl->relativeTolerance = 0.0;
        //we can either fill the x with zeros or cpy b to x;
        for (int i = 0; i < pX->num_values; i++)
            x[i] = b[i];

        return hcsparseSuccess;
    }


    //n == number of rows;
    const auto N = pA->num_cols;

    hcdenseVector y;
    hcdenseVector p;
    hcdenseVector r;
    hcdenseVector r_star;
    hcdenseVector s;
    hcdenseVector Mp;
    hcdenseVector AMp;
    hcdenseVector Ms;
    hcdenseVector AMs;

    y.values = am_alloc(sizeof(T)*N, acc, 0);
    p.values = am_alloc(sizeof(T)*N, acc, 0);
    r.values = am_alloc(sizeof(T)*N, acc, 0);
    r_star.values =am_alloc(sizeof(T)*N, acc, 0);
    s.values = am_alloc(sizeof(T)*N, acc, 0);
    Mp.values = am_alloc(sizeof(T)*N, acc, 0);
    AMp.values = am_alloc(sizeof(T)*N, acc, 0);
    Ms.values = am_alloc(sizeof(T)*N, acc, 0);
    AMs.values = am_alloc(sizeof(T)*N, acc, 0);

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

    one_buff[0] = 1;
    T zero_buff = 1;

    hcsparseScalar one;
    hcsparseScalar zero;

    one.value = am_alloc(sizeof(T)*1, acc, 0);
    zero.value = am_alloc(sizeof(T)*1, acc, 0);
    control->accl_view.copy(one_buff, one.value, sizeof(T)*1);
    control->accl_view.copy(&zero_buff, zero.value, sizeof(T)*1);

    one.offValue = 0;
    zero.offValue = 0;

    // y = A * x
    status = csrmv<T>(&one, pA, pX, &zero, &y, control);

#if 0
#ifndef NDEBUG
    float *x_h = (float *)calloc(N, sizeof(T));
    int *Arow = (int *)calloc(pA->num_rows+1, sizeof(int));
    int *Acol = (int *)calloc(pA->num_nonzeros, sizeof(int));
    T *Aval = (T *)calloc(pA->num_nonzeros, sizeof(T));
    float *y_h = (float *)calloc(N, sizeof(T));
    control->accl_view.copy(pX->values, x_h, N*sizeof(T));
    control->accl_view.copy(pA->values, Aval, pA->num_nonzeros*sizeof(T));
    control->accl_view.copy(pA->rowOffsets, Arow, (pA->num_rows+1)*sizeof(int));
    control->accl_view.copy(pA->colIndices, Acol, pA->num_nonzeros*sizeof(int));
    for (int i = 0; i < N ; i++) 
      std::cout << i << " x_h : " << x_h[i] << std::endl;
    for (int i = 0; i < pA->num_nonzeros ; i++) 
      std::cout << i << " val : " << Aval[i] << " col : " << Acol[i] << std::endl;
#endif
#endif

    // r = b - y
    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);

#if 0
#ifndef NDEBUG
    float *r_h = (float *)calloc(N, sizeof(T));
    float *B_h = (float *)calloc(N, sizeof(T));
    control->accl_view.copy(r.values, r_h, N*sizeof(T));
    control->accl_view.copy(y.values, y_h, N*sizeof(T));
    control->accl_view.copy(pB->values, B_h, N*sizeof(T));
    for (int i = 0; i < N ; i++) 
      std::cout << i << " r_h : " << r_h[i] << " pB : " 
                << B_h[i] << " y_h : " << y_h[i] << std::endl;
#endif
#endif

    T *norm_r_buff = (T*) calloc(1, sizeof(T));

    hcsparseScalar norm_r;
    norm_r.value = am_alloc(sizeof(T)*1, acc, 0);
    norm_r.offValue = 0;

    status = Norm1<T>(&norm_r, &r, control);
    control->accl_view.copy(norm_r.value, norm_r_buff, sizeof(T)*1);
    T *residuum_buff = (T*) calloc(1, sizeof(T));
    residuum_buff[0] = (T)(norm_r_buff[0] / norm_b_buff[0]);

    solverControl->initialResidual = residuum_buff[0];

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
    control->accl_view.copy(r.values, p.values, sizeof(T)*N);

    //Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
    control->accl_view.copy(r.values, r_star.values, sizeof(T)*N);

    // holder for <r_star, r>
    T *r_star_r_old_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar r_star_r_old;
    r_star_r_old.value = am_alloc(sizeof(T)*1, acc, 0); // &av_r_star_r_old;
    r_star_r_old.offValue = 0;

    // holder for <r_star, r_{i+1}>
    T *r_star_r_new_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar r_star_r_new;
    r_star_r_new.value = am_alloc(sizeof(T)*1, acc, 0); //&av_r_star_r_new;
    r_star_r_new.offValue = 0;

    status = dot<T>(&r_star_r_old, &r_star, &r, control);
    control->accl_view.copy(r_star_r_old.value, r_star_r_old_buff, sizeof(T)*1);

    int iteration = 0;
    bool converged = false;

    T *alpha_buff = (T*) calloc(1, sizeof(T));
    T *beta_buff = (T*) calloc(1, sizeof(T));
    T *omega_buff = (T*) calloc(1, sizeof(T));

    hcsparseScalar alpha;
    hcsparseScalar beta;
    hcsparseScalar omega;

    alpha.value = am_alloc(sizeof(T)*1, acc, 0); //&av_alpha;
    alpha.offValue = 0;
  
    beta.value = am_alloc(sizeof(T)*1, acc, 0); //&av_beta;
    beta.offValue = 0;
    
    omega.value = am_alloc(sizeof(T)*1, acc, 0); //&av_omega;
    omega.offValue = 0;

    // holder for <r_star, AMp>
    T *r_star_AMp_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar r_star_AMp;
    r_star_AMp.value = am_alloc(sizeof(T)*1, acc, 0); //&av_r_star_AMp;
    r_star_AMp.offValue = 0;

    // hoder for <A*M*s, s>
    T *AMsS_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar AMsS;
    AMsS.value = am_alloc(sizeof(T)*1, acc, 0); //&av_AMsS;
    AMsS.offValue = 0;

    // holder for <AMs, AMs>
    T *AMsAMs_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar AMsAMs;
    AMsAMs.value = am_alloc(sizeof(T)*1, acc, 0); //&av_AMsAMs;
    AMsAMs.offValue = 0;

    // holder for norm_s;
    T *norm_s_buff = (T*) calloc(1, sizeof(T));
    hcsparseScalar norm_s;
    norm_s.value = am_alloc(sizeof(T)*1, acc, 0); //&av_norm_s;
    norm_s.offValue = 0;

    while (!converged)
    {
        //Mp = M*p //apply preconditioner
        M(&p, &Mp, control);

        //AMp = A*Mp
        status = csrmv<T>(&one, pA, &Mp, &zero, &AMp, control);

        //<r_star, A*M*p>
        status = dot<T>(&r_star_AMp, &r_star, &AMp, control);
        control->accl_view.copy(r_star_AMp.value, r_star_AMp_buff, sizeof(T)*1);

        alpha_buff[0] = div<T>(r_star_r_old_buff[0], r_star_AMp_buff[0]);
        control->accl_view.copy(alpha_buff, alpha.value, sizeof(T)*1);

        //s_j = r - alpha*Mp
        status = axpby<T, EW_MINUS>(&s, &one, &r, &alpha, &AMp, control);

        status = Norm1<T>(&norm_s, &s, control);
        control->accl_view.copy(norm_s.value, norm_s_buff, sizeof(T)*1);

        residuum_buff[0] = div<T>(norm_s_buff[0], norm_b_buff[0]);

        if (solverControl->finished(residuum_buff[0]))
        {
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
        control->accl_view.copy(AMsS.value, AMsS_buff, sizeof(T)*1);
        control->accl_view.copy(AMsAMs.value, AMsAMs_buff, sizeof(T)*1);
        omega_buff[0] = div(AMsS_buff[0], AMsAMs_buff[0]);
        control->accl_view.copy(omega_buff, omega.value, sizeof(T)*1);

#ifndef NDEBUG
        if(omega_buff[0] == 0)
            std::cout << "omega = 0" ;
#endif

        //x = x + alpha*Mp + omega*Ms;
        status = axpy<T>(pX, &alpha, &Mp, pX, control);

        status = axpy<T>(pX, &omega, &Ms, pX, control);

        // r = s - omega * A*M*s
        status = axpy<T, EW_MINUS>(&r, &omega, &AMs, &s, control);

        status = Norm1<T>(&norm_r, &r, control);
        control->accl_view.copy(norm_r.value, norm_r_buff, sizeof(T)*1);
        residuum_buff[0] = div<T>(norm_r_buff[0], norm_b_buff[0]);

        if (solverControl->finished(residuum_buff[0]))
        {
            solverControl->nIters = iteration;
            break;
        }

        //beta = <r_star, r+1> / <r_star, r> * (alpha/omega)
        status = dot<T>(&r_star_r_new, &r_star, &r, control);
        control->accl_view.copy(r_star_r_new.value, r_star_r_new_buff, sizeof(T)*1);

        //TODO:: is it the best order?
        beta_buff[0] = div<T>(r_star_r_new_buff[0], r_star_r_old_buff[0]);
        beta_buff[0] = multi<T>(beta_buff[0], alpha_buff[0]);
        beta_buff[0] = div<T>(beta_buff[0], omega_buff[0]);
        control->accl_view.copy(beta_buff, beta.value, sizeof(T)*1);

        r_star_r_old_buff[0] = r_star_r_new_buff[0];
        control->accl_view.copy(r_star_r_new.value, r_star_r_old.value, sizeof(T)*1);

        //p = r + beta* (p - omega A*M*p);
        status = axpy<T>(&p, &beta, &p, &r, control); // p = beta * p + r;
        beta_buff[0] = multi<T>(beta_buff[0], omega_buff[0]);  // (beta*omega)
        control->accl_view.copy(beta_buff, beta.value, sizeof(T)*1);
        status = axpy<T,EW_MINUS>(&p, &beta, &AMp, &p, control);  // p = p - beta*omega*AMp;

        iteration++;
        solverControl->nIters = iteration;

        solverControl->print();
    }

    am_free(norm_b.value);
    free(norm_b_buff);
    am_free(y.values);
    am_free(p.values);
    am_free(r.values);
    am_free(r_star.values);
    am_free(s.values);
    am_free(Mp.values);
    am_free(AMp.values);
    am_free(Ms.values);
    am_free(AMs.values);
    am_free(one.value);
    am_free(zero.value);
    free(one_buff);
    am_free(norm_r.value);
    free(norm_r_buff);
    free(residuum_buff);
    am_free(r_star_r_old.value);
    free(r_star_r_old_buff);
    am_free(r_star_r_new.value);
    free(r_star_r_new_buff);
    free(alpha_buff);
    free(beta_buff);
    free(omega_buff);
    am_free(alpha.value);
    am_free(beta.value);
    am_free(omega.value);
    am_free(r_star_AMp.value);
    am_free(AMsS.value);
    am_free(AMsAMs.value);
    am_free(norm_s.value);
    free(r_star_AMp_buff);
    free(AMsS_buff);
    free(AMsAMs_buff);
    free(norm_s_buff);

    return hcsparseSuccess;
}

#endif //SOLVER_BICGSTAB_H_
