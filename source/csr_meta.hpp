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

#endif
