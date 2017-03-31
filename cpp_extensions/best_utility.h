#ifndef CPP_EXTENSIONS_BEST_UTILITY_H
#define CPP_EXTENSIONS_BEST_UTILITY_H

#include <iostream>
#include <cmath>
#include <cstdint>

#include "double_utils.h"

class BestUtility{
private:
    int mem_size;
public:
    double best_utility = -INFINITY;

    /* XXX: using pointer arrays of len(n_features). This uses a lot more memory than will be needed
            However, it is much faster than using a vector and calling push_back each time a new optimal
            feature is found. We could implement a low_mem version using vectors. */
    long *best_feat_idx;
    double *best_feat_threshold;
    uint8_t *best_feat_kind;
    int best_n_equiv;

    BestUtility(int const &n_features){
        this->best_feat_idx = new long[n_features];
        this->best_feat_threshold = new double[n_features];
        this->best_feat_kind = new uint8_t[n_features];
        this->best_n_equiv = 0;
        this->mem_size = n_features;
    }

    inline void add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind);
    inline void clear();
    inline void set_utility(double const &utility);
    inline bool operator >(double const& utility);
    inline bool operator <(double const& utility);
    inline bool operator ==(double const& utility);
};

inline void BestUtility::add_equivalent(long const &feature_idx, double const &threshold, uint8_t const &kind) {
    if(this->best_n_equiv == this->mem_size){
        std::cout << "Error: memory overflow in BestUtility" << std::endl;
    }
    this->best_feat_idx[this->best_n_equiv] = feature_idx;
    this->best_feat_threshold[this->best_n_equiv] = threshold;
    this->best_feat_kind[this->best_n_equiv] = kind;
    best_n_equiv ++;
}

inline void BestUtility::clear() {
    this->best_n_equiv = 0;
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
