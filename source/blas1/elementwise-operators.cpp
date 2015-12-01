#include "elementwise-operators.h"

const char* ElementWiseOperatorTrait<EW_PLUS>::operation = "OP_EW_PLUS";
const char* ElementWiseOperatorTrait<EW_MINUS>::operation = "OP_EW_MINUS";
const char* ElementWiseOperatorTrait<EW_MULTIPLY>::operation = "OP_EW_MULTIPLY";
const char* ElementWiseOperatorTrait<EW_DIV>::operation = "OP_EW_DIV";
const char* ElementWiseOperatorTrait<EW_MIN>::operation = "OP_EW_MIN";
const char* ElementWiseOperatorTrait<EW_MAX>::operation = "OP_EW_MAX";
const char* ElementWiseOperatorTrait<EW_DUMMY>::operation = "OP_EW_DUMMY";
