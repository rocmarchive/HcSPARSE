
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
T add (T a, T b) restrict(amp)
{
    return a + b;
}

template<typename T>
T sub (T a, T b) restrict(amp)
{
    return a - b;
}

template<typename T>
T multi (T a, T b) restrict(amp)
{
    return a * b;
}

template<typename T>
T div (T a, T b) restrict(amp)
{
    return a / b;
}

template<typename T>
T min (T a, T b) restrict(amp)
{
    return (a >= b) ? b : a;
}

template<typename T>
T max (T a, T b) restrict(amp)
{
    return (a >= b) ? a : b;
}

template<typename T>
T dummy (T a, T b) restrict(amp)
{
    return a;
}

template<typename T, ElementWiseOperator OP>
T operation(T a, T b) restrict(amp)
{
    if (OP == EW_PLUS)
        add<T>(a, b);    
    else if (OP == EW_MINUS)
        sub<T>(a, b);    
    else if (OP == EW_MULTIPLY)
        multi<T>(a, b);    
    else if (OP == EW_DIV)
        div<T>(a, b);    
    else if (OP == EW_MIN)
        min<T>(a, b);    
    else if (OP == EW_MAX)
        max<T>(a, b);    
    else
        dummy<T>(a, b);    
}
