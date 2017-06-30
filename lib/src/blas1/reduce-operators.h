#ifndef _REDUCE_OPERATORS_H_
#define _REDUCE_OPERATORS_H_

enum ReduceOperator
{
    RO_PLUS = 0,
    RO_SQR,
    RO_SQRT,
    RO_FABS,
    RO_DUMMY //does nothing
};

template <typename T>
T plus (T a, T b) __attribute__((hc, cpu))
{
    return a + b;
}

template <typename T>
T sqr (T a, T b) __attribute__((hc, cpu))
{
    return a + b * b;
}

template <typename T>
T fabs (T a, T b) __attribute__((hc, cpu))
{
    return a + hc::precise_math::fabsf((float)b);
}

template <typename T>
T sqr_root (T a) __attribute__((hc, cpu))
{
    return hc::precise_math::sqrtf((float)a);
}

template <typename T>
T reduce_dummy (T a) __attribute__((hc, cpu))
{
    return a;
}

template <typename T, ReduceOperator OP>
T reduceOperation (T a, T b) __attribute__((hc, cpu))
{
    if (OP == RO_PLUS)
        return plus<T>(a, b);
    else if (OP == RO_SQR)
        return sqr<T>(a, b);
    else if (OP == RO_FABS)
        return fabs<T>(a, b);
}

template <typename T, ReduceOperator OP>
T reduceOperation (T a) __attribute__((hc, cpu))
{
    if (OP == RO_SQRT)
        return sqr_root<T>(a);
    else
        return reduce_dummy<T>(a);
}
#endif
