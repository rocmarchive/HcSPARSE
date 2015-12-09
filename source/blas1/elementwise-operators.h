#ifndef _HCSPARSE_ELEMENTWISE_OPERATORS_HPP_
#define _HCSPARSE_ELEMENTWISE_OPERATORS_HPP_

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

template<typename T, ElementWiseOperator OP>
T operation (T a, T b);

#endif //ELEMENTWISE_HPP
