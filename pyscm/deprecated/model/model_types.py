#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2014 Alexandre Drouin

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
"""

import numpy as np

from .base import BaseModel

class ConjunctionModel(BaseModel):
    def predict_proba(self, X):
        predictions = np.ones(X.shape[0], np.float32)
        for a in self.binary_attributes:
            predictions *= a.classify(X)
        return predictions

    def __str__(self):
        return self._to_string(separator=" and ")


class DisjunctionModel(BaseModel):
    def predict_proba(self, X):
        predictions = np.ones(X.shape[0], dtype=np.float32)
        for a in self.binary_attributes:
            predictions *= 1.0 - a.classify(X) # Proportion of the voters that predict False in a
        return 1.0 - predictions

    def __str__(self):
        return self._to_string(separator=" or ")