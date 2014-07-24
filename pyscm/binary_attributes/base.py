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

from ..utils import _class_to_string


class BaseMetaBinaryAttribute(object):
    """
    A binary attribute mixin class

    Parameters:
    -----------
    example_dependencies: array_like, shape=(n_example_dependencies,), default=[]
            A list containing an identifier for each training example on which the attribute depends.
    """

    def __init__(self, example_dependencies=[]):
        self._example_dependencies = example_dependencies
        super(BaseMetaBinaryAttribute, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the binary attribute.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        positive_proportion: numpy_array, (n_examples,)
            Returns the number of binary attributes inside the Meta BinaryAttribute that predict the examples as
            belonging to the positive class. For Single BinaryAttributes, this simply corresponds to the class
            assigned by the binary attribute.
        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a binary attribute that is the inverse of the current binary attribute (self).
        For any example, the label attributed by self must be the opposite of the label attributed
        by the inverse of self.
        
        Returns:
        --------
        inverse: BinaryAttribute
            A binary attribute that is the inverse of self.
        """
        raise NotImplementedError()

    @property
    def example_dependencies(self):
        """
        Returns the example dependencies

        Returns:
        --------
        example_dependencies: array_like, shape=(n_example_dependencies,)
            A list containing an element of any type for each example on which the attribute depends.
        """
        return self._example_dependencies

    @example_dependencies.setter
    def example_dependencies(self, value):
        """
        Sets the example dependencies for a binary attribute. This is especially useful in the compression set setting.

        Parameters:
        -----------
        value: array_like, shape=(n_example_dependencies,)
            A list containing an element of any type for each example on which the attribute depends.
        """
        self._example_dependencies = value

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class BaseBinaryAttributeList(object):
    """
    A binary attribute list mixin class
    """

    def __init__(self):
        super(BaseBinaryAttributeList, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the binary attributes in the list.

        Parameters:
        -----------
        X: numpy_array, (n_examples, n_features)
            The feature vectors of examples to classify.

        Returns:
        --------
        attribute_classifications: numpy_array, (n_examples, n_binary_attributes)
            A matrix containing the labels assigned to each example by each binary attribute individually.
        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item_idx):
        raise NotImplementedError()


class SingleBinaryAttribute(BaseMetaBinaryAttribute):
    def __len__(self):
        return 1

class SingleBinaryAttributeList(BaseBinaryAttributeList):
    def __init__(self):
        for binary_attribute in self:
            if not isinstance(binary_attribute, SingleBinaryAttribute):
                raise ValueError("A list of single binary attributes can only contain single binary attributes.")

        super(SingleBinaryAttributeList, self).__init__()


class MetaBinaryAttribute(BaseMetaBinaryAttribute):
    def __len__(self):
        raise NotImplementedError()


class MetaBinaryAttributeList(BaseBinaryAttributeList):
    def __init__(self):
        if not hasattr(self, "cardinalities"):
            raise AttributeError("A binary meta attribute list must contain their cardinalities.")

        super(MetaBinaryAttributeList, self).__init__()