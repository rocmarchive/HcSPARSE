#pragma once
#ifndef _HC_CSR_META_H_
#define _HC_CSR_META_H_

#include "hcsparse.h"

#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif

static inline unsigned int flp2(unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

// Short rows in CSR-Adaptive are batched together into a single row block.
// If there are a relatively small number of these, then we choose to do
// a horizontal reduction (groups of threads all reduce the same row).
// If there are many threads (e.g. more threads than the maximum size
// of our workgroup) then we choose to have each thread serially reduce
// the row.
// This function calculates the number of threads that could team up
// to reduce these groups of rows. For instance, if you have a
// workgroup size of 256 and 4 rows, you could have 64 threads
// working on each row. If you have 5 rows, only 32 threads could
// reliably work on each row because our reduction assumes power-of-2.
static inline ulong numThreadsForReduction(const ulong num_rows)
{
#if defined(__INTEL_COMPILER)
    return 256 >> (_bit_scan_reverse(num_rows-1)+1);
#elif (defined(__clang__) && __has_builtin(__builtin_clz)) || \
      !defined(__clang) && \
      defined(__GNUG__) && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
    return (256 >> (8*sizeof(int)-__builtin_clz(num_rows-1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    unsigned long bit_returned;
    _BitScanReverse(&bit_returned, (num_rows-1));
    return 256 >> (bit_returned+1);
#else
    return flp2(256/num_rows);
#endif
}

void ComputeRowBlocks( ulong* rowBlocks, size_t& rowBlockSize, const int* rowDelimiters,
                       const int nRows, const int blkSize, const int blkMultiplier, const int rows_for_vector, const bool allocate_row_blocks = true )
{
    ulong* rowBlocksBase;
    int total_row_blocks = 1; // Start at one because of rowBlock[0]

    if (allocate_row_blocks)
    {
        rowBlocksBase = rowBlocks;
        *rowBlocks = 0;
        rowBlocks++;
    }
    ulong sum = 0;
    ulong i, last_i = 0;

    // Check to ensure nRows can fit in 32 bits
    if( (ulong)nRows > (ulong)std::pow( 2, ROWBITS ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present (%d bits) !", ROWBITS );
        return;
    }

    int consecutive_long_rows = 0;
    for( i = 1; i <= nRows; i++ )
    {
        int row_length = ( rowDelimiters[ i ] - rowDelimiters[ i - 1 ] );
        sum += row_length;

        // The following section of code calculates whether you're moving between
        // a series of "short" rows and a series of "long" rows.
        // This is because the reduction in CSR-Adaptive likes things to be
        // roughly the same length. Long rows can be reduced horizontally.
        // Short rows can be reduced one-thread-per-row. Try not to mix them.
        if ( row_length > 128 )
            consecutive_long_rows++;
        else if ( consecutive_long_rows > 0 )
        {
            // If it turns out we WERE in a long-row region, cut if off now.
            if (row_length < 32) // Now we're in a short-row region
                consecutive_long_rows = -1;
            else
                consecutive_long_rows++;
        }

        // If you just entered into a "long" row from a series of short rows,
        // then we need to make sure we cut off those short rows. Put them in
        // their own workgroup.
        if ( consecutive_long_rows == 1 )
        {
            // Assuming there *was* a previous workgroup. If not, nothing to do here.
            if( i - last_i > 1 )
            {
                if (allocate_row_blocks)
                {
                    *rowBlocks = ( (i - 1) << (64 - ROWBITS) );
                    // If this row fits into CSR-Stream, calculate how many rows
                    // can be used to do a parallel reduction.
                    // Fill in the low-order bits with the numThreadsForRed
                    if (((i-1) - last_i) > rows_for_vector)
                        *(rowBlocks-1) |= numThreadsForReduction((i - 1) - last_i);
                    rowBlocks++;
                }
                total_row_blocks++;
                last_i = i-1;
                sum = row_length;
            }
        }
        else if (consecutive_long_rows == -1)
        {
            // We see the first short row after some long ones that
            // didn't previously fill up a row block.
            if (allocate_row_blocks)
            {
                *rowBlocks = ( (i - 1) << (64 - ROWBITS) );
                if (((i-1) - last_i) > rows_for_vector)
                    *(rowBlocks-1) |= numThreadsForReduction((i - 1) - last_i);
                rowBlocks++;
            }
            total_row_blocks++;
            last_i = i-1;
            sum = row_length;
            consecutive_long_rows = 0;
         }

        // Now, what's up with this row? What did it do?

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WGBITS == workgroup ID
        if( ( i - last_i == 1 ) && sum > blkSize )
        {
            int numWGReq = static_cast< int >( std::ceil( (double)row_length / (blkMultiplier*blkSize) ) );

            // Check to ensure #workgroups can fit in WGBITS bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = ( numWGReq < (int)std::pow( 2, WGBITS ) ) ? numWGReq : (int)std::pow( 2, WGBITS );

            if (allocate_row_blocks)
            {
                for( int w = 1; w < numWGReq; w++ )
                {
                    *rowBlocks = ( (i - 1) << (64 - ROWBITS) );
                    *rowBlocks |= static_cast< ulong >( w );
                    rowBlocks++;
                }
                *rowBlocks = ( i << (64 - ROWBITS) );
                rowBlocks++;
            }
            total_row_blocks += numWGReq;
            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if( ( i - last_i > 1 ) && sum > blkSize )
        {
            i--; // This row won't fit, so back off one.
            if (allocate_row_blocks)
            {
                *rowBlocks = ( i << (64 - ROWBITS) );
                if ((i - last_i) > rows_for_vector)
                    *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
                rowBlocks++;
            }
            total_row_blocks++;
            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if( sum == blkSize )
        {
            if (allocate_row_blocks)
            {
                *rowBlocks = ( i << (64 - ROWBITS) );
                if ((i - last_i) > rows_for_vector)
                    *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
                rowBlocks++;
            }
            total_row_blocks++;
            last_i = i;
            sum = 0;
            consecutive_long_rows = 0;
        }
    }

    // If we didn't fill a row block with the last row, make sure we don't lose it.
    if ( allocate_row_blocks && (*(rowBlocks-1) >> (64 - ROWBITS)) != static_cast< ulong>(nRows) )
    {
        *rowBlocks = ( static_cast< ulong >( nRows ) << (64 - ROWBITS) );
        if ((nRows - last_i) > rows_for_vector)
            *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
        rowBlocks++;
    }
    total_row_blocks++;

    if (allocate_row_blocks)
    {
        size_t dist = std::distance( rowBlocksBase, rowBlocks );
        assert( (2 * dist) <= rowBlockSize );
        // Update the size of rowBlocks to reflect the actual amount of memory used
        // We're multiplying the size by two because the extended precision form of
        // CSR-Adaptive requires more space for the final global reduction.
        rowBlockSize = 2 * dist;
    }
    else
        rowBlockSize = 2 * total_row_blocks;
}

inline size_t ComputeRowBlocksSize( const int* rowDelimiters, const int nRows, const unsigned int blkSize,
                                    const unsigned int blkMultiplier, const unsigned int rows_for_vector )
{
    size_t rowBlockSize;
    ComputeRowBlocks( (ulong*)NULL, rowBlockSize, rowDelimiters, nRows, blkSize, blkMultiplier, rows_for_vector, false );
    return rowBlockSize;
}
#endif
