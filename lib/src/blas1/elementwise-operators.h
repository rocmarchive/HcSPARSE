#ifndef _ELEMENTWISE_OPERATORS_H_
#define _ELEMENTWISE_OPERATORS_H_

enum ElementWiseOperator
{
    EW_PLUS = 0,
    EW_MINUS,
    EW_MULTIPLY,
    EW_DIV,
    EW_MIN,
    EW_MAX,
    EW_DUMMY //does nothing
};

template<typename T>
T add (T a, T b) __attribute__((hc, cpu))
{
    return a + b;
}

template<typename T>
T sub (T a, T b) __attribute__((hc, cpu))
{
    return a - b;
}

template<typename T>
T multi (T a, T b) __attribute__((hc, cpu))
{
    return a * b;
}

template<typename T>
T div (T a, T b) __attribute__((hc, cpu))
{
    return a / b;
}

template<typename T>
T min (T a, T b) __attribute__((hc, cpu))
{
    return (a >= b) ? b : a;
}

template<typename T>
T max (T a, T b) __attribute__((hc, cpu))
{
    return (a >= b) ? a : b;
}

template<typename T>
T dummy (T a, T b) __attribute__((hc, cpu))
{
    return a;
}

template<typename T, ElementWiseOperator OP>
T operation(T a, T b) __attribute__((hc, cpu))
{
    if (OP == EW_PLUS)
        return add<T>(a, b);    
    else if (OP == EW_MINUS)
        return sub<T>(a, b);    
    else if (OP == EW_MULTIPLY)
        return multi<T>(a, b);    
    else if (OP == EW_DIV)
        return div<T>(a, b);    
    else if (OP == EW_MIN)
        return std::min<T>(a, b);    
    else if (OP == EW_MAX)
        return std::max<T>(a, b);    
    else
        return dummy<T>(a, b);  
}
#endif
 
