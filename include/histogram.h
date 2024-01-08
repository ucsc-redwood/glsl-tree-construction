#pragma once
#include <omp.h>

struct AgentRadixSortHistogram{
    enum{
        RADIX_BITS = 8,
        RADIX_DIGITS = 1 << RADIX_BITS,
        MAX_NUM_PASSES = (sizeof(unsigned int) * 8 + RADIX_BITS - 1) / RADIX_BITS,
    };

    struct TempStorage{
        unsigned int bins[MAX_NUM_PASSES][RADIX_DIGITS];
    };

    TempStorage& s;

    int* bins_out;

    const int *keys;

    int num_items;

    int begin_bit, end_bit;

    int num_passes;

    AgentRadixSortHistogram(TempStorage& s, int* bins_out, const int* keys, int num_items, int begin_bit, int end_bit)
        : s(s), bins_out(bins_out), keys(keys), num_items(num_items), begin_bit(begin_bit), end_bit(end_bit)
    {
        num_passes = (end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS;
    }
    
    void Init(){
        #pragma  omp parallel for
        for(int i = 0; i < num_passes; ++i){
            #pragma unroll
            for(int j = 0; j < RADIX_DIGITS; ++j){
                s.bins[i][j] = 0;
            }
        }
    }

    void AccumulateSharedHistograms(){
        for (int current_bit = begin_bit, pass = 1; current_bit < end_bit; current_bit += RADIX_BITS, ++pass){
            #pragma omp parallel for
            for(int i = 0; i < num_items; ++i){
                int key = keys[i];
                int bin = (key >> current_bit) & (RADIX_DIGITS - 1);
                #pragma omp atomic
                s.bins[(current_bit - begin_bit) / RADIX_BITS][bin]++;
            }
        }
    }
    void AccumulateGlobalHistograms(){
        #pragma omp parallel for
        for(int bin = 0; bin < RADIX_DIGITS; ++bin){
            #pragma unroll
            
        }


    }


    void Process(){
        Init();
        A
        AccumulateGlobalHistograms();
    }
};