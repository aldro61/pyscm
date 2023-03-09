#include <cmath>
#include <iostream>

#include "best_utility.h"
#include "solver.h"


/*
 * Solver
 */

void get_n_examples_by_class(const bool* example_is_included, const long* y, const int &n_examples, double &n_negative, double &n_positive, double* sample_weight){
    for(int i = 0; i < n_examples; i++){
        if(example_is_included[i]){
            if(y[i] == 0){
                n_negative = n_negative + sample_weight[i];
            }
            else{
                n_positive = n_positive + sample_weight[i];
            }
        }
    }
}

void update_optimal_solution(BestUtility &best_solution, int const &feature_idx, double const &threshold,
                             double const &N, double const &P_bar, double const &p, double const &feature_weight,
                             double const &n_negative, double const &n_positive){
    // Get utility for x > t and check if optimal
    double utility_0 = (N - p * P_bar) * feature_weight;
    if(best_solution < utility_0){
        best_solution.clear();
        best_solution.set_utility(utility_0);
        best_solution.add_equivalent(feature_idx, threshold, 0, N, P_bar);
    } else if(best_solution == utility_0){
        best_solution.add_equivalent(feature_idx, threshold, 0, N, P_bar);
    }

    // Get utility for x <= t and check if optimal
    double N_1 = n_negative - N;
    double P_bar_1 = n_positive - P_bar;
    double utility_1 = ( N_1 - p *  P_bar_1) * feature_weight;
    if(best_solution < utility_1){
        best_solution.clear();
        best_solution.set_utility(utility_1);
        best_solution.add_equivalent(feature_idx, threshold, 1, N_1, P_bar_1);
    } else if(best_solution == utility_1){
        best_solution.add_equivalent(feature_idx, threshold, 1, N_1, P_bar_1);
    }
}

int find_max(double p,
             double *X,
             long *y,
             long *Xas,
             double *sample_weight,
             long *example_idx,
             double *feature_weights,
             int n_examples_included,
             int n_examples,
             int n_features,
             BestUtility &out_best_solution){

    // Make a mask that tells us which examples should be considered in the utility calculations
    bool *example_is_included = new bool[n_examples];
    std::fill_n(example_is_included, n_examples, false);
    
    for(int i = 0; i < n_examples_included; i++){
        example_is_included[example_idx[i]] = true;
    }

    // Find the number of positive and negative examples
    double n_negative = 0, n_positive = 0;
    get_n_examples_by_class(example_is_included, y, n_examples, n_negative, n_positive, sample_weight);

    // Utility calculations start
    for(int i = 0; i < n_features; i++){

        // For each threshold of this feature (a threshold is an example's feature value)
        double P_bar, prev_P_bar, N, prev_N;
        double prev_threshold;

        prev_N = 0;
        prev_P_bar = 0;
        prev_threshold = -INFINITY;
        for(int j = 0; j < n_examples; j++){

            // Get the index of the next example according to the sorting
            long idx = Xas[i * n_examples + j];

            // Consider this example only if it is included in the calculations
            if(example_is_included[idx]){

                // Get the example's label and threshold
                long label = y[idx];
                double threshold = X[idx * n_features + i];

                // Wait for the last example with this threshold before computing the utilities
                if(prev_threshold != -INFINITY && not_equal(threshold, prev_threshold)){
                    update_optimal_solution(out_best_solution, i, prev_threshold, N, P_bar, p, feature_weights[i],
                                            n_negative, n_positive);
                }

                if(label == 1){
                    P_bar = prev_P_bar + sample_weight[idx];
                    N = prev_N;
                }
                else{
                    P_bar = prev_P_bar;
                    N = prev_N + sample_weight[idx];
                }

                prev_N = N;
                prev_P_bar = P_bar;
                prev_threshold = threshold;
            }
        }
        update_optimal_solution(out_best_solution, i, prev_threshold, N, P_bar, p, feature_weights[i],
                                n_negative, n_positive);
    }
    delete [] example_is_included;
    return 0;
}
