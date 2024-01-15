#pragma once
#define RADIX_BIN 256
#define RADIX_LOG 8
#define RADIX_BITS 8
#define RADIX_MASK 255 // Mask of digit bins
#define RADIX_DIGITS 1 << RADIX_BITS
#define RADIX_PASS (sizeof(unsigned int) * 8 + RADIX_BITS - 1) / RADIX_BITS

#define LANE_COUNT 32 // number of threads in a subgroup
#define LANE_MASK 31
#define LANE_LOG 5

#define HIST_SUBGRUP                                                           \
  2 // number of subgroups in a thread block/work group for executing histogram
    // kernel
#define HIST_THREADS                                                           \
  64 // number of threads in a thread block/work group for executing histogram
     // kernel
#define HIST_TBLOCKS                                                           \
  2048 // number of thread blocks/workgroups for executing histogram kernel
#define HIST_BLOCK                                                             \
  HIST_TBLOCKS / HIST_THREADS // number of blocks for executing histogram kernel

#define HIST_PART_SIZE (input_size / HIST_TBLOCKS) // the size of each partition
#define HIST_PART_START                                                        \
  (gl_WorkGroupID * HIST_PART_SIZE) // the start index of each partition. work
                                    // group id in vulkan, block idx in cuda

#define HIST_PART_END                                                          \
  (gl_WorkGroupID == HIST_TBLOCKS - 1 ? input_size                             \
                                      : (gl_WorkGroupID + 1) * HIST_PART_SIZE)

int input_size;

//For the binning
#define BIN_PART_SIZE       7680    //The partition tile size of a BinningPass threadblock
#define BIN_HISTS_SIZE      4096    //The total size of all subgroup histograms in shared memory
#define BIN_TBLOCKS         512     //The number of threadblocks dispatched in a BinningPass threadblock
#define BIN_THREADS         512     //The number of threads in a BinningPass threadblock
#define BIN_SUB_PART_SIZE   480     //The subpartition tile size of a single subgroup in a BinningPass threadblock
#define BIN_SUBGROUPS       16      //The number of subgroup in a BinningPass threadblock
#define BIN_KEYS_PER_THREAD 15      //The number of keys per thread in BinningPass threadblock

#define BIN_PARTITIONS     (input_size / BIN_PART_SIZE)             //The number of partition tiles in a BinningPass
#define BIN_SUB_PART_START (SUBGROUP_IDX * BIN_SUB_PART_SIZE)     //The starting offset of a subpartition tile
#define BIN_PART_START     (partitionIndex * BIN_PART_SIZE)     //The starting offset of a partition tile


#define LANE gl_SubgroupInvocationID // the idx of thread in the subgroup
#define SUBGROUP_IDX gl_SubgroupID // the idx of subgroup the thread belongs to
#define SUBGROUP_THREAD_IDX                                                             \
  (LANE + (SUBGROUP_IDX << LANE_LOG)) // the subgroup relative thread idx


#define FLAG_NOT_READY      0       //Flag value inidicating neither inclusive sum, or aggregate sum of a partition tile is ready
#define FLAG_AGGREGATE      1       //Flag value indicating aggregate sum of a partition tile is ready
#define FLAG_INCLUSIVE      2       //Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK           3       //Mask used to retrieve flag values



#[global]
uint *b_sort;            // buffer to be sorted
#[global]
uint *b_alt;            // double buffer
#[global]               
uint *b_globalHist;     // buffer holding device level offsets for each
                           // binning pass

globallycoherent RWBuffer<uint> b_index;                        //buffer used to atomically assign partition tiles to threadblocks
globallycoherent RWBuffer<uint> b_passHist;                     //buffer used to store reduced sums of partition tiles
globallycoherent RWBuffer<uint> b_passTwo;                      //buffer used to store reduced sums of partition tiles
globallycoherent RWBuffer<uint> b_passThree;                    //buffer used to store reduced sums of partition tiles
globallycoherent RWBuffer<uint> b_passFour;                     //buffer used to store reduced sums of partition tiles

#[groupshared]
uint g_globalHist[RADIX_BIN][RADIX_PASS]; //Shared memory for performing the upfront global histogram
#[groupshared]
uint g_localHist[RADIX_BIN];  //Threadgroup copy of globalHist during digit binning passes
#[groupshared]
uint g_subgroupHists[BIN_PART_SIZE];                    //Shared memory for the per subgroup histograms during digit binning passes
#[groupshared]
uint g_reductionHist[RADIX_BIN];                        //Shared memory for reduced per subgroup histograms during digit binning pass


void GlobalHistogram() {
  // initialization
  for (int i = SUBGROUP_THREAD_IDX; i < RADIX_BIN; i += HIST_THREADS) {
    g_globalHist[i][0] = 0;
    g_globalHist[i][1] = 0;
    g_globalHist[i][2] = 0;
    g_globalHist[i][3] = 0;
    work_group_sync();
  }

  // histogram
  // add number of occurence of each 1 at different digit place to global histogram
  // there are 8 digits place for each pass, so we have 8*4 digits place in total for 32 bits integer
  const int partitionEnd = HIST_PART_END;
  for (int i = SUBGROUP_THREAD_IDX + HIST_PART_START; i < partitionEnd;
       i += HIST_THREADS) {
    const uint key = b_sort[i];
    atomic_add(g_globalHist[key & RADIX_MASK][0], 1);
    atomic_add(g_globalHist[key >> RADIX_LOG & RADIX_MASK][1], 1);
    atomic_add(g_globalHist[key >> (2 * RADIX_LOG) & RADIX_MASK][2], 1);
    atomic_add(g_globalHist[key >> (3 * RADIX_LOG) & RADIX_MASK][3], 1);
  }
  work_group_sync();

  // prefix_sum
  // prefix sum at warp/subgroup/wave level
  // scan occurence of digits containg digit 1 in the first i+1 bins for each warp/subgroup/wave, and store the result in g_globalHist
  for (int i = SUBGROUP_IDX << LANE_LOG; i < RADIX_BIN; i += HIST_THREADS) {
    g_globalHist[((LANE + 1) & LANE_MASK) + i] =
        SubGroupPrefixSum(g_globalHist[LANE + i]) + g_globalHist[LANE + i];
  }
    work_group_sync();

  if (LANE < (RADIX_BIN >> LANE_LOG) && SUBGROUP_IDX == 0) {
    g_globalHist[LANE << LANE_LOG] += SubGroupPrefixSum(g_globalHist[LANE << LANE_LOG]);
  }
    work_group_sync();

    int k = SUBGROUP_THREAD_IDX;
    
    // prefixsum at global level
    // b_globalHist holds global bin offset, it is used to indicate where each block can begin scattering its keys into the different digit bins
    // for example, b_globalHist[255] = 100 meanning that there are 100 keys that contain digit 1 in the first 255 bins(least signifcant 8 bits)
  
    atomic_add(b_globalHist[k], (LANE ? g_globalHist[k].x : 0) + (SUBGROUP_IDX ? SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][0], 0) : 0));
    atomic_add(b_globalHist[k + RADIX_BIN], (LANE ? g_globalHist[k].y : 0) + (SUBGROUP_IDX ? SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][1], 0) : 0));
    atomic_add(b_globalHist[k + RADIX_BIN*2], (LANE ? g_globalHist[k].z : 0) + (SUBGROUP_IDX ? SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][2], 0) : 0));
    atomic_add(b_globalHist[k + RADIX_BIN*3], (LANE ? g_globalHist[k].w : 0) + (SUBGROUP_IDX ? SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][3], 0) : 0));

    for (k += HIST_THREADS; k < RADIX_BIN; k += HIST_THREADS)
    {
        atomic_add(b_globalHist[k], (LANE ? g_globalHist[k][0] : 0) + SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][0], 0));
        atomic_add(b_globalHist[k + RADIX_BIN], (LANE ? g_globalHist[k][1] : 0) + SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][1], 0));
        atomic_add(b_globalHist[k + RADIX_BIN*2], (LANE ? g_globalHist[k][2] : 0) + SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][2], 0));
        atomic_add(b_globalHist[k + RADIX_BIN*3], (LANE ? g_globalHist[k][3] : 0) + SubgroupReadLaneAt(g_globalHist[k - LANE_COUNT][3], 0));
    }
}


void BinningPass()
{
    //load the global histogram values into shared memory
    if (SUBGROUP_THREAD_IDX < RADIX_BIN)
        g_localHist[SUBGROUP_THREAD_IDX] = b_globalHist[SUBGROUP_THREAD_IDX];
    
    //atomically fetch and increment device memory index to assign partition tiles
    //Take advantage of the barrier to also clear the shared memory
    // g_localHist[0] is set to be the original value of b_index[0]
    if (LANE == 0 && SUBGROUP_IDX == 0)
        atomic_add(b_index[0], 1, g_localHist[0]);
    work_group_sync();
    // Returns the value of the local histogram at bin 0 for the given lane index within the specified subgroup.
    // it returns the current partitionIndex 
    int partitionIndex = SubgroupReadLaneAt(g_localHist[0], 0);
    for (int i = SUBGROUP_THREAD_IDX; i < BIN_HISTS_SIZE; i += BIN_THREADS)
        g_subGroupHists[i] = 0;
    work_group_sync();
    
    //Load keys into registers
    uint keys[BIN_KEYS_PER_THREAD];
    {
        [unroll]
        for (int i = 0, t = LANE + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = b_sort[t];
    }

    // [ 2, 5, 6 , 1]
    // output= [1, 2, 3, 0]
    // [0, 2, 7, 13, 14]
    //Warp Level Multisplit
    // this step ranks the key in warp-level by computes both 
    // (1) a warp-wide histogram of digit-counts, and (2) the warp-relative digit prefix count for each key
    
    uint offsets[BIN_KEYS_PER_THREAD];
    {
        const uint t = SUBGROUP_IDX << RADIX_LOG;
            
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {   
            // fill with 1 so we can do ascending sort
            offsets[i] = 0xFFFFFFFF;

                [unroll]
            for (int k = 0; k < RADIX_LOG; ++k)
            {
                // extracts the k-th bit of the key, t2 is true if the k-th bit of keys[i] is 1, and false otherwise.
                const bool t2 = keys[i] >> k & 1;
                // subgroupActiveBallot counts the number of thread in the subgroup that have the k-th bit of keys[i] set to 1
                // if t2 is true, then 0 xor subgroupActiveBallot(t2), oherwise 0xFFFFFFFF xor subgroupActiveBallot(t2), we get inverse result for t2 = true and false
                offsets[i] &= (t2 ? 0 : 0xFFFFFFFF) ^ subgroupActiveBallot(t2);
            }
            
            // counts the number of bits the current lane is responsible for, i.e. in a subgroup of 32 threads, 
            // if the current lane id is 0, then it is responsible for 1 bits, if it is 1, then it is responsible for 2 bits, and so on
            const uint bits = countbits(offsets[i] << LANE_MASK - LANE);
            // index = least significant 8 bits of keys[i] + subgroup id * radix bin
            const int index = (keys[i] & RADIX_MASK) + t;
            const uint prev = g_subgroupHists[index];
            //the lowest ranked thread in each active digit
            // population will  add the population’s count to the per-warp counter corresponding to that digit
            if (bits == 1)
                g_subgroupHists[index] += countbits(offsets[i]);
            // updated to reflect the position where the i-th key should be placed in the sorted array.
            offsets[i] = prev + bits - 1;
        }
    }
    work_group_sync();
        
    // block level exclusive prefix sum across the histograms
    // This provides (1) tile-wide digit counts for participating in chained scan cooperation with other blocks, and 
    // (2) tile-relative bin offsets for each warp to scatter its elements into local bins within shared memory.
    if (SUBGROUP_THREAD_IDX < RADIX_BIN)
    {
        for (int k = SUBGROUP_THREAD_IDX + RADIX_BIN; k < BIN_HISTS_SIZE; k += RADIX_BIN)
        {
            g_subgroupHists[SUBGROUP_THREAD_IDX] += g_subgroupHists[k];
            g_subgroupHists[k] = g_subgroupHists[SUBGROUP_THREAD_IDX] - g_subgroupHists[k];
        }
        g_reductionHist[SUBGROUP_THREAD_IDX] = g_subgroupHists[SUBGROUP_THREAD_IDX];
        
        // partitionIndex == 0 means it is the first partition, so  the value is the inclusive global prefix sum of the digit counts of this and all previous tiles
        if (partitionIndex == 0)
            atomic_add(b_passHist[SUBGROUP_THREAD_IDX * BIN_PARTITIONS + partitionIndex], FLAG_INCLUSIVE ^ (g_subgroupHists[SUBGROUP_THREAD_IDX] << 2));
        else
        // otherwsie,  value is the local, per-tile digit count
            atomic_add(b_passHist[SUBGROUP_THREAD_IDX * BIN_PARTITIONS + partitionIndex], FLAG_AGGREGATE ^ (g_subgroupHists[SUBGROUP_THREAD_IDX] << 2));
    }
    work_group_sync(); // a barrier must be placed between here and the lookback to prevent accidentally broadcasting an incorrect aggregate

    //exclusive prefix sum across the reductions
    if (SUBGROUP_THREAD_IDX < RADIX_BIN)
        g_reductionHist[(LANE + 1 & LANE_MASK) + (SUBGROUP_IDX << LANE_LOG)] = SubgroupPrefixSum(g_reductionHist[SUBGROUP_THREAD_IDX]) + g_reductionHist[SUBGROUP_THREAD_IDX];
    work_group_sync();
        
    if (LANE < (RADIX_BIN >> LANE_LOG) && SUBGROUP_IDX == 0)
        g_reductionHist[LANE << LANE_LOG] = SubgroupPrefixSum(g_reductionHist[LANE << LANE_LOG]);
    work_group_sync();

    if (SUBGROUP_THREAD_IDX < RADIX_BIN && LANE)
        g_reductionHist[SUBGROUP_THREAD_IDX] += SubgroupReadLaneAt(g_reductionHist[SUBGROUP_THREAD_IDX - 1], 1);
    work_group_sync();
    //
    //Update offsets
    if (SUBGROUP_IDX)
    {
        const uint t = SUBGROUP_IDX << RADIX_LOG;
            
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            const uint t2 = keys[i] & RADIX_MASK;
            offsets[i] += g_subgroupHists[t2 + t] + g_reductionHist[t2];
        }
    }
    else
    {
        [unroll]
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += g_reductionHist[keys[i] & RADIX_MASK];
    }
    work_group_sync();
        
    //Scatter keys into shared memory
    // now the g_subgroupHists contain the offset of each scattered key in the subgroup
    {
        for (int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            g_subgroupHists[offsets[i]] = keys[i];
    }
    work_group_sync();
        
    //Lookback
    // block 31 scan = block 30 scan result + current block prefix sum
    // block 31 = block 29 scan result + block 30 prefix sum + current block prefix sum
     
    if (partitionIndex)
    {
        //Free up shared memory, because we are at max
        // set t to be scattered key.
        const uint t = g_subgroupHists[SUBGROUP_IDX];
        // for loop at block level
        for (int i = SUBGROUP_IDX; i < RADIX_BIN; i += BIN_SUBGROUPS)
        {
            uint aggregate = 0;

            // set the prefix sum for current subgroup to be the partitionIndex
            if (LANE == 0)
                g_subgroupHists[SUBGROUP_IDX] = partitionIndex;
            // Within the same block, look backward progressively to find the predecessors that record the global inclusive prefix up to and including current tile.
            for (int k = partitionIndex - LANE - 1; 0 <= k;)
            {
                // get the reduction sum on the predecessors bin in current block
                uint flagPayload = b_passHist[i * BIN_PARTITIONS + k];
                // if the reduction sum on the predecessors bin is ready for all threads in current subgroup
                if (subGroupActiveAllTrue((flagPayload & FLAG_MASK) > FLAG_NOT_READY))
                {   
                    // if the predecessors record the global inclusive prefix up to and including current tile
                    if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
                    {   
                        // if current lane is the first lane in the subgroup, then simply set the value to k as index k indicates that  represent the global inclusive prefix sum
                        if (subGroupIsFirstLane())
                            g_subgroupHists[SUBGROUP_IDX] = k;
                    }
                    
                    // if the prefix sum of current subgroup is less than partitionIndex, which means we set g_subgroupHists[SUBGROUP_IDX] to be k in the previous if statement
                    // which means the current subgroup hist record the global inclusive prefix up to and including current tile.
                    if (g_subgroupHists[SUBGROUP_IDX] < partitionIndex)
                    {
                        // sum up the prefix sum of each lane in the subgroup that is ready (which we set to be k in the previous if statement)
                        aggregate += subGroupActiveSum(k >= g_subgroupHists[SUBGROUP_IDX] ? (flagPayload >> 2) : 0);
                        
                        // if current lane is the first lane in the subgroup
                        if (LANE == 0)
                        {
                            //  then we  add the aggregate sum to the reduction sum of current block and end the lookback for current block
                            atomic_add(b_passHist[i * BIN_PARTITIONS + partitionIndex], 1 ^ (aggregate << 2));
                            // if the subgroup is not the first subgroup, then we add the aggregate sum to the global prefix sum until the previous subgroup
                            g_reductionHist[i] = aggregate + (i ? g_localHist[i] : 0) - g_reductionHist[i];
                        }
                        break;
                    }
                    // if the prefix sum of current subgroup is greater or equal to partitionIndex,
                    // which means the current subgroup hist contains the aggregate prefix sum of the k-th bin in current block
                    // 
                    else
                    {
                        aggregate += subGroupActiveSum(flagPayload >> 2);
                        k -= LANE_COUNT;
                    }
                }
            }
        }
            
        //place value back
        if (LANE == 0)
            g_subgroupHists[SUBGROUP_IDX] = t;
    }
    else
    {
        if (SUBGROUP_THREAD_IDX < RADIX_BIN)
            g_reductionHist[SUBGROUP_THREAD_IDX] = (SUBGROUP_THREAD_IDX ? g_localHist[SUBGROUP_THREAD_IDX] : 0) - g_reductionHist[SUBGROUP_THREAD_IDX];
    }
    work_group_sync();
        
    //Scatter runs of keys into device memory;
    {
        for (int i = SUBGROUP_THREAD_IDX; i < BIN_PART_SIZE; i += BIN_THREADS)
            b_alt[g_reductionHist[g_subgroupHists[i] & RADIX_MASK] + i] = g_subgroupHists[i];
    }
        
    //for input sizes which are not perfect multiples of the partition tile size
    if (partitionIndex == BIN_PARTITIONS - 1)
    {
        if (SUBGROUP_THREAD_IDX < RADIX)
            g_reductionHist[SUBGROUP_THREAD_IDX] = (b_passHist[SUBGROUP_THREAD_IDX * BIN_PARTITIONS + partitionIndex] >> 2) + (SUBGROUP_THREAD_IDX ? g_localHist[SUBGROUP_THREAD_IDX] : 0);
        work_group_sync();
        
        partitionIndex++;
        for (int i = SUBGROUP_THREAD_IDX + BIN_PART_START; i < e_size; i += BIN_THREADS)
        {
            const uint key = b_sort[i];
            uint offset = 0xFFFFFFFF;
            
            [unroll]
            for (int k = 0; k < RADIX_LOG; ++k)
            {
                const bool t = key >> k & 1;
                offset &= (t ? 0 : 0xFFFFFFFF) ^ WaveActiveBallot(t);
            }
            
            [unroll]
            for (int k = 0; k < BIN_SUBGROUPS; ++k)
            {
                if (SUBGROUP_IDX == k)
                {
                    const uint t = g_reductionHist[key & RADIX_MASK];
                    if (countbits(offset << LANE_MASK - LANE) == 1)
                        g_reductionHist[key & RADIX_MASK] += countbits(offset);
                    offset = t + countbits((offset << LANE_MASK - LANE) << 1);
                }
                work_group_sync();
            }

            b_alt[offset] = key;
        }
    }

}



void SortComputeArray(int size, int *keys) {}

void work_group_sync() {
  // do group sync
}

void atomic_add(uint *dest, uint *value) {}

void prefix_sum(uint **input){}

// read input at lane idx
void SubgroupReadLaneAt(uint *input, int idx) {}