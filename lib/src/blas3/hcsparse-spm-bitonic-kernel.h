template <typename T>
hcsparseStatus compute_nnzC_Ct_bitonic_scan_32 (int num_threads, 
                                             int num_blocks,
                                             int j, int position, 
                                             int *queue_one, 
                                             int *csrRowPtrA, 
                                             int *csrColIndA, 
                                             T *csrValA, 
                                             int *csrRowPtrB, 
                                             int *csrColIndB, 
                                             T *csrValB, 
                                             int *csrRowPtrC, 
                                             int *csrRowPtrCt, 
                                             int *csrColIndCt, 
                                             T *csrValCt, 
                                             int n, 
                                             hcsparseControl* control)
{
    size_t szLocalWorkSize;
    size_t szGlobalWorkSize;
    
    szLocalWorkSize  = num_threads;
    szGlobalWorkSize = num_blocks * szLocalWorkSize;

        int numStages = 0;
        for(int temp = 2 * num_threads; temp > 1; temp >>= 1)
        {
            ++numStages;
        }
        std::cout << "numStages : " << numStages << std::endl;

    std::cout << "num_blocks = " << num_blocks << std::endl;

    hc::extent<1> grdExt(szGlobalWorkSize);
    hc::tiled_extent<1> t_ext = grdExt.tile(szLocalWorkSize);
    hc::parallel_for_each(control->accl_view, t_ext, [=] (hc::tiled_index<1> &tidx) [[hc]]
    {
        tile_static int s_key[2*256];
        tile_static T s_val[2*256];
        tile_static int s_scan[2*256 + 1];
        int local_id = tidx.local[0];
        int group_id = tidx.tile[0];
        int global_id = tidx.global[0];
        int local_size = tidx.tile_dim[0];
        int width = local_size * 2;
        int i, local_counter = 0;
        int strideB, local_offset, global_offset;
        int invalid_width;
        int local_id_halfwidth = local_id + local_size;
        int row_id_B; // index_type
        int row_id;// index_type
        row_id = queue_one[TUPLE_QUEUE * (position + group_id)];
        int start_col_index_A, stop_col_index_A;  // index_type
        int start_col_index_B, stop_col_index_B;  // index_type
        T value_A;                            // value_type
        start_col_index_A = csrRowPtrA[row_id];
        stop_col_index_A  = csrRowPtrA[row_id + 1];
        // i is both col index of A and row index of B
        for (i = start_col_index_A; i < stop_col_index_A; i++)
        {
            row_id_B = csrColIndA[i];
            value_A  = csrValA[i];
            start_col_index_B = csrRowPtrB[row_id_B];
            stop_col_index_B  = csrRowPtrB[row_id_B + 1];
            strideB = stop_col_index_B - start_col_index_B;
            if (local_id < strideB)
            {
                local_offset = local_counter + local_id;
                global_offset = start_col_index_B + local_id;
                s_key[local_offset] = csrColIndB[global_offset];
                s_val[local_offset] = csrValB[global_offset] * value_A;
            }
            if (local_id_halfwidth < strideB)
            {
                local_offset = local_counter + local_id_halfwidth;
                global_offset = start_col_index_B + local_id_halfwidth;
                s_key[local_offset] = csrColIndB[global_offset];
                s_val[local_offset] = csrValB[global_offset] * value_A;
            }
            local_counter += strideB;
        }
        tidx.barrier.wait();

        invalid_width = width - local_counter;
        // to meet 2^N, set the rest elements to n (number of columns of C)
        if (local_id < invalid_width)
            s_key[local_counter + local_id] = n;
        tidx.barrier.wait();

        // bitonic sort
        /* BITONIC SCAN STARTS */
        int numStages = 0;
        for(int temp = width; temp > 1; temp >>= 1)
        {
            ++numStages;
        }
        for (int stage = 0; stage < numStages; ++stage)
        {
            for (int passOfStage = 0; passOfStage <= stage; ++passOfStage)
            {
              int sortIncreasing = 1;
              int pairDistance = 1 << (stage - passOfStage);
              int blockWidth   = 2 * pairDistance;
              int leftId = (local_id % pairDistance) + (local_id / pairDistance) * blockWidth;
              int rightId = leftId + pairDistance;
              int sameDirectionBlockWidth = 1 << stage;
              if((local_id/sameDirectionBlockWidth) % 2 == 1)
                  sortIncreasing = 1 - sortIncreasing;
              int  leftElement  = s_key[leftId];          // index_type
              int  rightElement = s_key[rightId];         // index_type
              T   leftElement_val  = s_val[leftId];      // value_type
              T   rightElement_val = s_val[rightId];     // value_type
              int  greater;         // index_type
              int  lesser;          // index_type
              T   greater_val;     // value_type
              T   lesser_val;      // value_type
              if(leftElement > rightElement)
              {
                  greater = leftElement;
                  lesser  = rightElement;
                  greater_val = leftElement_val;
                  lesser_val  = rightElement_val;
              }
              else
              {
                  greater = rightElement;
                  lesser  = leftElement;
                  greater_val = rightElement_val;
                  lesser_val  = leftElement_val;
              }
              if(sortIncreasing)
              {
                  s_key[leftId]  = lesser;
                  s_key[rightId] = greater;
                  s_val[leftId]  = lesser_val;
                  s_val[rightId] = greater_val;
              }
              else
              {
                  s_key[leftId]  = greater;
                  s_key[rightId] = lesser;
                  s_val[leftId]  = greater_val;
                  s_val[rightId] = lesser_val;
              }
           }
        }
        tidx.barrier.wait();

        /* BITONIC SCAN ENDS */

        // compression - scan
        /* COMPRESSION SCAN STARTS */

        bool duplicate = 1;
        bool duplicate_halfwidth = 1;
        // generate bool value in registers
        if (local_id < local_counter && local_id > 0)
            duplicate = (s_key[local_id] != s_key[local_id - 1]);
        if (local_id_halfwidth < local_counter)
            duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);
        // copy bool values from register to local memory (s_scan)
        s_scan[local_id]                    = duplicate;
        s_scan[local_id_halfwidth]          = duplicate_halfwidth;
        tidx.barrier.wait();

        int ai, bi;
        int baseai = 1 + 2 * local_id;
        int basebi = baseai + 1;
        int temp;
    
        ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
        tidx.barrier.wait();

        if (local_size == 64) {
          if (local_id == 0) { s_scan[127] = s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
          s_scan[63] = 0;
          tidx.barrier.wait();
        }
        
#if 0
          if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
          tidx.barrier.wait();
          if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
        } else if (local_size == 64) {
          if (local_id < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          tidx.barrier.wait();
          if (local_id < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
          if (local_id == 0) { s_scan[127] += s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
          if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 16) { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          if (local_id < 32) { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
          tidx.barrier.wait();
        }
#endif
        ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
        tidx.barrier.wait();

#if 0
        // compute final position and final value in registers
        int   move_pointer;
        int   final_position, final_position_halfwidth;
        int   final_key,      final_key_halfwidth;
        T final_value,    final_value_halfwidth;
            final_position = s_scan[local_id];
            final_position_halfwidth = s_scan[local_id_halfwidth];
            final_key = s_scan[local_id]; //s_key[local_id];
            final_key_halfwidth = s_scan[local_id_halfwidth]; //s_key[local_id_halfwidth];

        if (local_id < local_counter && duplicate == 1)
        {
            final_position = s_scan[local_id];
            final_key = s_key[local_id];
            final_value = s_val[local_id];
            move_pointer = local_id + 1;
            while (s_scan[move_pointer] == s_scan[move_pointer + 1])
            {
                final_value += s_val[move_pointer];
                move_pointer++;
            }
        }
        if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
        {
            final_position_halfwidth = s_scan[local_id_halfwidth];
            final_key_halfwidth = s_key[local_id_halfwidth];
            final_value_halfwidth = s_val[local_id_halfwidth];
            move_pointer = local_id_halfwidth + 1;
            while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
            {
                final_value_halfwidth += s_val[move_pointer];
                move_pointer++;
            }
        }
        tidx.barrier.wait();

        // write final_positions and final_values to s_key and s_val
        if (local_id < local_counter && duplicate == 1)
        {
            s_key[final_position] = final_key;
            s_val[final_position] = final_value;
        }
        if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
        {
            s_key[final_position_halfwidth] = final_key_halfwidth;
            s_val[final_position_halfwidth] = final_value_halfwidth;
        }
        tidx.barrier.wait();
        /* COMPRESSION SCAN ENDS*/
    
        local_counter = s_scan[width] - invalid_width;
#endif
        if (local_id == 0)
            csrRowPtrC[row_id] = 0; //local_counter;
#if 1
        // write compressed lists to global mem
        int row_offset = queue_one[TUPLE_QUEUE * (position + group_id) + 1];
        if (local_id < local_counter)
        {
            global_offset = row_offset + local_id;
            csrColIndCt[global_offset] = s_scan[local_id]; //s_key[local_id];
            csrValCt[global_offset] = s_val[local_id];
        }
        if (local_id_halfwidth < local_counter)
        {
            global_offset = row_offset + local_id_halfwidth;
            csrColIndCt[global_offset] = s_scan[local_id_halfwidth]; //s_key[local_id_halfwidth];
            csrValCt[global_offset] = s_val[local_id_halfwidth];
        }
#endif
    
    }).wait();
    
    return hcsparseSuccess;
}
