[numthreads(LANE_COUNT, BIN_WAVES, 1)]
void FourthBinningPass(int3 gtid : SV_GroupThreadID, int3 gid : SV_GroupID)
{
    //load the global histogram values into shared memory
    if (GROUP_THREAD_ID < RADIX)
        g_localHist[GROUP_THREAD_ID] = b_globalHist[GROUP_THREAD_ID + FOURTH_RADIX_START];
    
    //atomically fetch and increment device memory index to assign partition tiles
    //Take advantage of the barrier to also clear the shared memory
    if (LANE == 0 && WAVE_INDEX == 0)
        InterlockedAdd(b_index[3], 1, g_localHist[0]);
    GroupMemoryBarrierWithGroupSync();
    int partitionIndex = WaveReadLaneAt(g_localHist[0], 0);
    for (int i = GROUP_THREAD_ID; i < BIN_HISTS_SIZE; i += BIN_THREADS)
        g_waveHists[i] = 0;
    GroupMemoryBarrierWithGroupSync();
        
    //Load keys into registers
    uint keys[BIN_KEYS_PER_THREAD];
    {
        [unroll]
        for (int i = 0, t = LANE + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = b_alt[t];
    }

    //Warp Level Multisplit
    uint offsets[BIN_KEYS_PER_THREAD];
    {
        const uint t = WAVE_INDEX << RADIX_LOG;
            
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            offsets[i] = 0xFFFFFFFF;

            [unroll]
            for (int k = FOURTH_RADIX; k < 32; ++k)
            {
                const bool t2 = keys[i] >> k & 1;
                offsets[i] &= (t2 ? 0 : 0xFFFFFFFF) ^ WaveActiveBallot(t2);
            }
                
            const uint bits = countbits(offsets[i] << LANE_MASK - LANE);
            const int index = (keys[i] >> FOURTH_RADIX) + t;
            const uint prev = g_waveHists[index];
            if (bits == 1)
                g_waveHists[index] += countbits(offsets[i]);
            offsets[i] = prev + bits - 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();
        
    //exclusive prefix sum across the histograms
    if (GROUP_THREAD_ID < RADIX)
    {
        for (int k = GROUP_THREAD_ID + RADIX; k < BIN_HISTS_SIZE; k += RADIX)
        {
            g_waveHists[GROUP_THREAD_ID] += g_waveHists[k];
            g_waveHists[k] = g_waveHists[GROUP_THREAD_ID] - g_waveHists[k];
        }
        g_reductionHist[GROUP_THREAD_ID] = g_waveHists[GROUP_THREAD_ID];
            
        if (partitionIndex == 0)
            InterlockedAdd(b_passFour[GROUP_THREAD_ID * BIN_PARTITIONS + partitionIndex], FLAG_INCLUSIVE ^ (g_waveHists[GROUP_THREAD_ID] << 2));
        else
            InterlockedAdd(b_passFour[GROUP_THREAD_ID * BIN_PARTITIONS + partitionIndex], FLAG_AGGREGATE ^ (g_waveHists[GROUP_THREAD_ID] << 2));
    }
    GroupMemoryBarrierWithGroupSync(); // a barrier must be placed between here and the lookback to prevent accidentally broadcasting an incorrect aggregate

    //exclusive prefix sum across the reductions
    if (GROUP_THREAD_ID < RADIX)
        g_reductionHist[(LANE + 1 & LANE_MASK) + (WAVE_INDEX << LANE_LOG)] = WavePrefixSum(g_reductionHist[GROUP_THREAD_ID]) + g_reductionHist[GROUP_THREAD_ID];
    GroupMemoryBarrierWithGroupSync();
        
    if (LANE < (RADIX >> LANE_LOG) && WAVE_INDEX == 0)
        g_reductionHist[LANE << LANE_LOG] = WavePrefixSum(g_reductionHist[LANE << LANE_LOG]);
    GroupMemoryBarrierWithGroupSync();

    if (GROUP_THREAD_ID < RADIX && LANE)
        g_reductionHist[GROUP_THREAD_ID] += WaveReadLaneAt(g_reductionHist[GROUP_THREAD_ID - 1], 1);
    GroupMemoryBarrierWithGroupSync();
        
    //Update offsets
    if (WAVE_INDEX)
    {
        const uint t = WAVE_INDEX << RADIX_LOG;
            
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            const uint t2 = keys[i] >> FOURTH_RADIX;
            offsets[i] += g_waveHists[t2 + t] + g_reductionHist[t2];
        }
    }
    else
    {
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += g_reductionHist[keys[i] >> FOURTH_RADIX];
    }
    GroupMemoryBarrierWithGroupSync();
        
    //Scatter keys into shared memory
    //For some bizarre reason, this is significantly faster rolled
    {
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            g_waveHists[offsets[i]] = keys[i];
    }
    GroupMemoryBarrierWithGroupSync();
        
    //Lookback
    if (partitionIndex)
    {
        //Free up shared memory, because we are at max
        const uint t = g_waveHists[WAVE_INDEX];
            
        for (int i = WAVE_INDEX; i < RADIX; i += BIN_WAVES)
        {
            uint aggregate = 0;
            if (LANE == 0)
                g_waveHists[WAVE_INDEX] = partitionIndex;
                
            for (int k = partitionIndex - LANE - 1; 0 <= k;)
            {
                uint flagPayload = b_passFour[i * BIN_PARTITIONS + k];
                if (WaveActiveAllTrue((flagPayload & FLAG_MASK) > FLAG_NOT_READY))
                {
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {
                        if (WaveIsFirstLane())
                            g_waveHists[WAVE_INDEX] = k;
                    }
                        
                    if (g_waveHists[WAVE_INDEX] < partitionIndex)
                    {
                        aggregate += WaveActiveSum(k >= g_waveHists[WAVE_INDEX] ? (flagPayload >> 2) : 0);
                                
                        if (LANE == 0)
                        {
                            InterlockedAdd(b_passFour[i * BIN_PARTITIONS + partitionIndex], 1 ^ (aggregate << 2));
                            g_reductionHist[i] = aggregate + (i ? g_localHist[i] : 0) - g_reductionHist[i];
                        }
                        break;
                    }
                    else
                    {
                        aggregate += WaveActiveSum(flagPayload >> 2);
                        k -= LANE_COUNT;
                    }
                }
            }
        }
            
        //place value back
        if (LANE == 0)
            g_waveHists[WAVE_INDEX] = t;
    }
    else
    {
        if (GROUP_THREAD_ID < RADIX)
            g_reductionHist[GROUP_THREAD_ID] = (GROUP_THREAD_ID ? g_localHist[GROUP_THREAD_ID] : 0) - g_reductionHist[GROUP_THREAD_ID];
    }
    GroupMemoryBarrierWithGroupSync();
        
    //Scatter runs of keys into device memory;
    {
        for (int i = GROUP_THREAD_ID; i < BIN_PART_SIZE; i += BIN_THREADS)
            b_sort[g_reductionHist[g_waveHists[i] >> FOURTH_RADIX] + i] = g_waveHists[i];
    }
    
    //for input sizes which are not perfect multiples of the partition tile size
    if (partitionIndex == BIN_PARTITIONS - 1)
    {
        if (GROUP_THREAD_ID < RADIX)
            g_reductionHist[GROUP_THREAD_ID] = (b_passFour[GROUP_THREAD_ID * BIN_PARTITIONS + partitionIndex] >> 2) + (GROUP_THREAD_ID ? g_localHist[GROUP_THREAD_ID] : 0);
        GroupMemoryBarrierWithGroupSync();
        
        partitionIndex++;
        for (int i = GROUP_THREAD_ID + BIN_PART_START; i < e_size; i += BIN_THREADS)
        {
            const uint key = b_alt[i];
            uint offset = 0xFFFFFFFF;
            
            [unroll]
            for (int k = FOURTH_RADIX; k < 32; ++k)
            {
                const bool t = key >> k & 1;
                offset &= (t ? 0 : 0xFFFFFFFF) ^ WaveActiveBallot(t);
            }
            
            [unroll]
            for (int k = 0; k < BIN_WAVES; ++k)
            {
                if (WAVE_INDEX == k)
                {
                    const uint t = g_reductionHist[key >> FOURTH_RADIX];
                    if (countbits(offset << LANE_MASK - LANE) == 1)
                        g_reductionHist[key >> FOURTH_RADIX] += countbits(offset);
                    offset = t + countbits((offset << LANE_MASK - LANE) << 1);
                }
                GroupMemoryBarrierWithGroupSync();
            }

            b_sort[offset] = key;
        }
    }
}