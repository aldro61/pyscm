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

class BaseModel(object):
    def __init__(self):
        self.binary_attributes = []
        super(BaseModel, self).__init__()

    def add(self, binary_attribute):
        self.binary_attributes.append(binary_attribute)

    def predict(self, X):
        predictions = self.predict_proba(X)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        return np.asarray(predictions, dtype=np.uint8)

    def predict_proba(self, X):
        raise NotImplementedError()

    def remove(self, index):
        del self.binary_attributes[index]

    @property
    def example_dependencies(self):
        return [d for ba in self.binary_attributes for d in ba.example_dependencies]

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.binary_attributes])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        for ba in self.binary_attributes:
            yield ba

    def __len__(self):
        return len(self.binary_attributes)

    def __str__(self):
        return self._to_string()