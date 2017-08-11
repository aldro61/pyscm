#ifndef CPP_EXTENSIONS_BEST_UTILITY_H
#define CPP_EXTENSIONS_BEST_UTILITY_H

#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>

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
    int *best_N;
    int *best_P_bar;
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
        delete [] this->best_N;
        delete [] this->best_P_bar;
    }

    inline void add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind,
                                int const &N, int const &P_bar);
    inline void clear();
    inline void resize(int const &memory_size);
    inline void set_utility(double const &utility);
    inline bool operator >(double const& utility);
    inline bool operator <(double const& utility);
    inline bool operator ==(double const& utility);
};

inline void BestUtility::add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind,
                                        int const &N, int const &P_bar) {
    if(this->best_n_equiv == this->mem_size){
        // We need to resize the array
        this->resize((this->mem_size > 1 ? this->mem_size : 2) * MEM_RESIZE_INCREASE_FACTOR);
    }
    this->best_feat_idx[this->best_n_equiv] = feature_idx;
    this->best_feat_threshold[this->best_n_equiv] = threshold;
    this->best_feat_kind[this->best_n_equiv] = kind;
    this->best_N[this->best_n_equiv] = N;
    this->best_P_bar[this->best_n_equiv] = P_bar;
    best_n_equiv ++;
}

inline void BestUtility::clear() {
    this->best_n_equiv = 0;
}

inline void BestUtility::resize(int const &memory_size) {
    long *best_feat_idx_new = new long[memory_size];
    double *best_feat_threshold_new = new double[memory_size];
    uint8_t* best_feat_kind_new = new uint8_t[memory_size];
    int *best_N_new = new int[memory_size];
    int *best_P_bar_new = new int[memory_size];

    // Copy the data
    bool copied = false;
    if (this->best_n_equiv >= 1){
        copied = true;
        std::copy_n(this->best_feat_idx, this->best_n_equiv, best_feat_idx_new);
        std::copy_n(this->best_feat_threshold, this->best_n_equiv, best_feat_threshold_new);
        std::copy_n(this->best_feat_kind, this->best_n_equiv, best_feat_kind_new);
        std::copy_n(this->best_N, this->best_n_equiv, best_N_new);
        std::copy_n(this->best_P_bar, this->best_n_equiv, best_P_bar_new);

    }
    if(copied){
        delete [] this->best_feat_idx;
        delete [] this->best_feat_threshold;
        delete [] this->best_feat_kind;
        delete [] this->best_N;
        delete [] this->best_P_bar;
    }

    this->best_feat_idx = best_feat_idx_new;
    this->best_feat_threshold = best_feat_threshold_new;
    this->best_feat_kind = best_feat_kind_new;
    this->mem_size = memory_size;
    this->best_N = best_N_new;
    this->best_P_bar = best_P_bar_new;
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