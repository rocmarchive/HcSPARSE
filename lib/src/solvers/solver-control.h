#include "hcsparse.h"
#include "preconditioners/diagonal.h"

hcsparseSolverControl*
hcsparseCreateSolverControl(PRECONDITIONER precond, int maxIters,
                            double relTol, double absTol)
{
    hcsparseSolverControl *solver_control = new hcsparseSolverControl();

    if(!solver_control)
    {
        solver_control = nullptr;
    }

    solver_control->absoluteTolerance = absTol;
    solver_control->relativeTolerance = relTol;
    solver_control->nIters = 0;
    solver_control->maxIters = maxIters;
    solver_control->initialResidual = 0;
    solver_control->preconditioner = precond;

    return solver_control;

}

hcsparseStatus
hcsparseReleaseSolverControl(hcsparseSolverControl *solverControl)
{

    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    solverControl->absoluteTolerance = -1;
    solverControl->relativeTolerance = -1;
    solverControl->nIters = -1;
    solverControl->maxIters = -1;
    solverControl->initialResidual = -1;
    solverControl->preconditioner = NOPRECOND;

    delete solverControl;

    solverControl = nullptr;
    return hcsparseSuccess;
}

// set the solver control parameters for next use;
hcsparseStatus
hcsparseSetSolverParams(hcsparseSolverControl *solverControl,
                        PRECONDITIONER precond,
                        int maxIters, double relTol, double absTol)
{
    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    solverControl->absoluteTolerance = absTol;
    solverControl->relativeTolerance = relTol;
    solverControl->nIters = 0;
    solverControl->maxIters = maxIters;
    solverControl->initialResidual = 0;
    solverControl->preconditioner = precond;

    return hcsparseSuccess;

}

hcsparseStatus
hcsparseSolverPrintMode(hcsparseSolverControl *solverControl, PRINT_MODE mode)
{
    if (solverControl == nullptr)
    {
        return hcsparseInvalid;
    }

    solverControl->printMode = mode;

    return hcsparseSuccess;
}
