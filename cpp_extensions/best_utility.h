#ifndef CPP_EXTENSIONS_BEST_UTILITY_H
#define CPP_EXTENSIONS_BEST_UTILITY_H

#include <iostream>
#include <cmath>
#include <cstdint>
#include <string.h>

#include "double_utils.h"

#define MEM_RESIZE_INCREASE_FACTOR 2

class BestUtility{
private:
    int mem_size;
public:
    double best_utility;
    long *best_feat_idx;
    double *best_feat_threshold;
    uint8_t *best_feat_kind;
    int best_n_equiv;

    BestUtility(int const &memory_size){
        this->best_utility = - INFINITY;
        this->best_n_equiv = 0;
        this->resize(memory_size);
    }
    
    ~BestUtility(){
        delete [] this->best_feat_idx;
        delete [] this->best_feat_threshold;
        delete [] this->best_feat_kind;
    }

    inline void add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind);
    inline void clear();
    inline void resize(int const &memory_size);
    inline void set_utility(double const &utility);
    inline bool operator >(double const& utility);
    inline bool operator <(double const& utility);
    inline bool operator ==(double const& utility);
};

inline void BestUtility::add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind) {
    if(this->best_n_equiv == this->mem_size){
        // We need to resize the array
        this->resize((this->mem_size > 1 ? this->mem_size : 2) * MEM_RESIZE_INCREASE_FACTOR);
    }
    this->best_feat_idx[this->best_n_equiv] = feature_idx;
    this->best_feat_threshold[this->best_n_equiv] = threshold;
    this->best_feat_kind[this->best_n_equiv] = kind;
    best_n_equiv ++;
}

inline void BestUtility::clear() {
    this->best_n_equiv = 0;
}

inline void BestUtility::resize(int const &memory_size) {
    long *best_feat_idx_new = new long[memory_size];
    double *best_feat_threshold_new = new double[memory_size];
    uint8_t* best_feat_kind_new = new uint8_t[memory_size];

    // Copy the data
    bool copied = false;
    if (this->best_n_equiv >= 1){
        copied = true;
        memcpy(best_feat_idx_new, this->best_feat_idx, this->best_n_equiv*sizeof(long));
        memcpy(best_feat_threshold_new, this->best_feat_threshold, this->best_n_equiv*sizeof(double));
        memcpy(best_feat_kind_new, this->best_feat_kind, this->best_n_equiv*sizeof(uint8_t));
    }
    if(copied){
        delete [] this->best_feat_idx;
        delete [] this->best_feat_threshold;
        delete [] this->best_feat_kind;
    }

    this->best_feat_idx = best_feat_idx_new;
    this->best_feat_threshold = best_feat_threshold_new;
    this->best_feat_kind = best_feat_kind_new;
    this->mem_size = memory_size;
}

inline void BestUtility::set_utility(double const &utility) {
    this->best_utility = utility;
}

inline bool BestUtility::operator>(double const &utility) {
    return greater(this->best_utility, utility);
}

inline bool BestUtility::operator<(double const &utility) {
    return less(this->best_utility, utility);
}

inline bool BestUtility::operator==(double const &utility) {
    return equal(utility, this->best_utility);
}

#endif //CPP_EXTENSIONS_BEST_UTILITY_H
