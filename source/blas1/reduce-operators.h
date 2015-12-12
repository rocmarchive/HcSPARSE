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

template<typename T>
T plus (T a, T b) restrict(amp)
{
    return a + b;
}

template<typename T>
T sqr (T a, T b) restrict(amp)
{
    return a + b * b;
}

template<typename T>
T sqrt (T a, T b) restrict(amp)
{
    return a + Concurrency::fast_math::sqrt(b);
}

template<typename T>
T fabs (T a, T b) restrict(amp)
{
    return a + Concurrency::fast_math::fabs(b);
}

template<typename T>
T reduce_dummy (T a, T b) restrict(amp)
{
    return a;
}

template<typename T, ReduceOperator OP>
T reduceOperation (T a, T b) restrict(amp)
{
    if (OP == RO_PLUS)
        return plus<T>(a, b);
    else if (OP == RO_SQR)
        return sqr<T>(a, b);
    else if (OP == RO_SQRT)
        return sqrt<T>(a, b);
    else if (OP == RO_FABS)
        return fabs<T>(a, b);
    else
        return reduce_dummy<T>(a, b);
}
#endif
