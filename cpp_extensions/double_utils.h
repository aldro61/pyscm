#ifndef CPP_EXTENSIONS_DOUBLE_UTILS_H_H
#define CPP_EXTENSIONS_DOUBLE_UTILS_H_H

#include <cmath>
#include <iostream>

#define TOL 1e-7

inline bool equal(double x, double y){
    return fabs(x - y) <= TOL;
}

inline bool not_equal(double x, double y){
    return fabs(x - y) > TOL;
}

inline bool greater(double x, double y){
    if(fabs(x) == INFINITY || fabs(y) == INFINITY){
        return x > y;
    }
    else{
        return fabs(x - y) > TOL && x > y;
    }
}

inline bool less(double x, double y){
    if(fabs(x) == INFINITY || fabs(y) == INFINITY){
        return x < y;
    }
    else{
        return fabs(x - y) > TOL && x < y;
    }
}

#endif //CPP_EXTENSIONS_DOUBLE_UTILS_H_H
