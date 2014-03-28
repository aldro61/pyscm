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

from .base import ModelMixin


class ConjunctionModel(ModelMixin):
    def predict(self, X):
        predictions = np.ones(X.shape[0], np.uint8)
        for a in self.binary_attributes:
            predictions = np.logical_and(predictions, a.classify(X))
        return np.asarray(predictions, dtype=np.uint8)

    def __str__(self):
        return self._to_string(separator=" and ")


class DisjunctionModel(ModelMixin):
    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=np.uint8)
        for a in self.binary_attributes:
            predictions = np.logical_or(predictions, a.classify(X))
        return np.asarray(predictions, dtype=np.uint8)

    def __str__(self):
        return self._to_string(separator=" or ")