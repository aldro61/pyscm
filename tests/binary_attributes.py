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
import sys
import unittest

sys.path.append("../..")

from pyscm.binary_attributes import DecisionStump

class TestDecisionStump(unittest.TestCase):
    def test_classify(self):
        X = np.array([[1, 2],
                      [2, 3]])

        self.assertTrue(np.all(DecisionStump(0, -1, 0).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, -1, 1).classify(X) == np.array([1, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, -1, 2).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 0).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 1).classify(X) == np.array([0, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 2).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, -1, 0).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, -1, 1).classify(X) == np.array([1, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, -1, 2).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 0).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 1).classify(X) == np.array([0, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(0, 1, 2).classify(X) == np.array([0, 0], dtype=np.int8)))

        self.assertTrue(np.all(DecisionStump(1, -1, 0).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, -1, 2).classify(X) == np.array([1, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, -1, 3).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 0).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 2).classify(X) == np.array([0, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 3).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, -1, 0).classify(X) == np.array([0, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, -1, 2).classify(X) == np.array([1, 0], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, -1, 3).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 0).classify(X) == np.array([1, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 2).classify(X) == np.array([0, 1], dtype=np.int8)))
        self.assertTrue(np.all(DecisionStump(1, 1, 3).classify(X) == np.array([0, 0], dtype=np.int8)))

    def test_inverse(self):
        stump = DecisionStump(0, 1, 42)
        stump_inverse = stump.inverse()

        self.assertEqual(stump.feature_idx, stump_inverse.feature_idx)
        self.assertEqual(stump.threshold, stump_inverse.threshold)
        self.assertEquals(stump_inverse.direction, stump.direction * -1)

        X = np.array([[42]])
        self.assertEquals(stump.classify(X)[0], False)
        self.assertEquals(stump_inverse.classify(X)[0], True)