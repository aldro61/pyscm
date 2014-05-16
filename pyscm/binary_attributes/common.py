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

from .base import BinaryAttributeListMixin


class DefaultBinaryAttributeList(BinaryAttributeListMixin):
    """
    A binary attribute list that can be used with any type of binary attributes.

    Parameters:
    -----------
    binary_attributes: list, len=n_binary_attributes
        A list of unique binary attributes to be used to build the model.

    Note:
    -----
    This type of list works for any binary attribute, although using lists specially designed for a type of binary
    attribute can significantly reduce memory usage. This is achieved by using lazy generation of the binary attributes,
    as some binary attributes can be defined by a few parameters which are far less expensive to store than instances of
    classes.
    """

    def __init__(self, binary_attributes):
        self.binary_attributes = binary_attributes
        BinaryAttributeListMixin.__init__(self)

    def __len__(self):
        return len(self.binary_attributes)

    def __getitem__(self, item_idx):
        return self.binary_attributes[item_idx]

    def classify(self, X):
        """
        Classifies a set of examples using the binary attributes in the list.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of the examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_binary_attributes)
            List of labels assigned to each example classified according to their binary attributes.
        """
        attribute_classifications = np.zeros((X.shape[0], len(self)), dtype=np.uint8)
        for i, ba in enumerate(self.binary_attributes):
            attribute_classifications[:, i] = ba.classify(X)
        return attribute_classifications


