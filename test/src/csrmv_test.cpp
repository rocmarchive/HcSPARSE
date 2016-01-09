#include <iostream>
#include "../hcsparse.h"
#include <hc.hpp>
using namespace hc;

int main(int argc, char *argv[])
{
    double alpha = 1.0;
    double beta = 1.0;
    bool extended_precision = false;
    bool no_zeroes = false;
    std::string filePath = "sample1.mtx";

    hcsparseScalar halpha;
    hcsparseCsrMatrix matx;
    hcdenseVector x;
    hcsparseScalar hbeta;
    hcdenseVector y;

    hcsparseStatus status;

    status = hcsparseSetup();
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseSetup() failed "<<std::endl;

    accelerator acc = accelerator(accelerator::default_accelerator);

    // Override the default CPU access type.
    acc.set_default_cpu_access_type(access_type_read_write);

    // Create an accelerator_view from the default accelerator. The
    // accelerator_view reflects the default_cpu_access_type of the
    // accelerator it’s associated with.
    accelerator_view acc_v = acc.get_default_view();

    hcsparseControl control(acc_v);

    int n_vals, n_rows, n_cols;
    status = hcsparseHeaderfromFile( &n_vals, &n_rows, &n_cols, filePath.c_str( ) );

    status = hcsparseInitCsrMatrix( &matx );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseInitCsrMatrix( matx ) failed "<<std::endl;

    matx.num_rows = n_rows;
    matx.num_cols = n_cols;
    matx.num_nonzeros = n_vals;

    array_view<double ,1> arr(matx.num_nonzeros);
    matx.values = &arr;

    array_view<int ,1> arr1(matx.num_nonzeros);
    matx.colIndices = &arr1;

    array_view<int ,1> arr2(matx.num_rows + 1);
    matx.rowOffsets = &arr2;

    status = hcsparseDCsrMatrixfromFile( &matx, (const char*)filePath.c_str( ), &control, no_zeroes );
    if( status != hcsparseSuccess )
        std::cout << "Could not read matrix market data from disk: "<< filePath.c_str() << std::endl;

    hcsparseCsrMetaSize( &matx, &control );

    array_view<ulong ,1> arr0(matx.rowBlockSize);
    matx.rowBlocks =  &arr0;

    hcsparseCsrMetaCompute( &matx, &control );

    status = hcsparseInitScalar( &halpha );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseInitScalar( alpha ) failed "<<std::endl;

    array_view<double ,1> arr3(1);
    arr3[0] = alpha;
    halpha.value = &arr3;

    status = hcsparseInitScalar( &hbeta );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseInitScalar( beta ) failed "<<std::endl;

    array_view<double ,1> arr4(1);
    arr4[0] = beta;
    hbeta.value = &arr4;

    status = hcsparseInitVector( &x );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseInitVector( x ) failed "<<std::endl;

    array_view<double ,1> arr5(n_cols);
    x.values = &arr5;

    x.num_values = n_cols;

    status = hcsparseInitVector( &y );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseInitVector( y ) failed "<<std::endl;

    array_view<double ,1> arr6(n_rows);
    y.values = &arr6;

    y.num_values = n_rows;

    status = hcsparseScsrmv(&halpha, &matx, &x, &hbeta, &y, &control );
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseScsrmv failed "<<std::endl;

    array_view<double ,1> *ydum = static_cast<array_view<double ,1 > *>(y.values);

    ydum->synchronize();

    status = hcsparseTeardown();
    if ( status != hcsparseSuccess )
        std::cout << " hcsparseTeardown() "<<std::endl;

    return 1;
}
