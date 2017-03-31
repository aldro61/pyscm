/*
pyscm -- The Set Covering Machine in Python
Copyright (C) 2017 Alexandre Drouin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */
#ifndef CPP_EXTENSIONS_UTILITY_H
#define CPP_EXTENSIONS_UTILITY_H

#include <vector>

#include "best_utility.h"

int find_max(double p,
             double *X,
             long *y,
             long *Xas,
             long *example_idx,
             double *feature_weights,
             int n_examples_included, // examples that we are allowed to look at
             int n_examples,
             int n_features,
             BestUtility &out_best_solution);


#endif //CPP_EXTENSIONS_UTILITY_H
