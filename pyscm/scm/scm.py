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
from functools import partial
from math import ceil

import numpy as np

from .base import BaseSetCoveringMachine
from ..model import ConjunctionModel, DisjunctionModel, conjunction, disjunction
from ..binary_attributes.base import SingleBinaryAttributeList, MetaBinaryAttributeList
from ..utils import _conditional_print, _split_into_contiguous
from .popcount import inplace_popcount_32, inplace_popcount_64


def _column_sum_dtype(array):
    if array.shape[0] <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif array.shape[0] <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif array.shape[0] <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return dtype


def _block_sum_rows(row_idx, array, n_examples, row_block_size=1, column_block_size=1000, verbose=False):
    _verbose_print = partial(_conditional_print, condition=verbose)
    # Builds a mask to turn off the bits of the rows we do not want to count in the sum.
    def build_row_mask(example_idx, n_examples, mask_n_bits):
        if mask_n_bits not in [8, 16, 32, 64, 128]:
            raise ValueError("Unsupported mask format. Use 8, 16, 32, 64 or 128 bits.")

        n_masks = int(ceil(float(n_examples) / mask_n_bits))
        masks = [0] * n_masks

        for idx in example_idx:
            example_mask = idx / mask_n_bits
            example_mask_idx = mask_n_bits - (idx - mask_n_bits * example_mask) - 1
            masks[example_mask] |= 1 << example_mask_idx

        return np.array(masks, dtype="u" + str(mask_n_bits / 8))

    # Get the size of the ints used to store the data
    if array.dtype == np.uint32:
        n_bits = 32
        inplace_popcount = inplace_popcount_32
    elif array.dtype == np.uint64:
        n_bits = 64
        inplace_popcount = inplace_popcount_64
    else:
        raise ValueError("Unsupported data type for compressed attribute classifications array. The supported data" +
                         " types are np.uint32 and np.uint64.")

    mask = build_row_mask(row_idx, n_examples, n_bits)
    rows_to_load = np.where(mask != 0)[0]  # Don't load ints if we don't need their bits.
    sum_res = np.zeros(array.shape[1], dtype=_column_sum_dtype(row_idx))

    n_row_blocks = int(ceil(float(len(rows_to_load)) / row_block_size))
    n_col_blocks = int(ceil(float(array.shape[1]) / column_block_size))

    row_count = 0
    for row_block in xrange(n_row_blocks):
        row_block_start_idx = row_block * row_block_size
        row_block_stop_idx = row_block_start_idx + row_block_size
        row_mask = mask[rows_to_load[row_block_start_idx: row_block_stop_idx]]

        for col_block in xrange(n_col_blocks):
            col_block_start_idx = col_block * column_block_size
            col_block_stop_idx = col_block_start_idx + column_block_size
            block = array[rows_to_load[row_block_start_idx: row_block_stop_idx],
                    col_block_start_idx: col_block_stop_idx]

            if len(block.shape) == 1:
                block = block.reshape(1, -1)

            if hasattr(block, "flags") and not block.flags["OWNDATA"]:
                block = block.copy()

            inplace_popcount(block, row_mask)
            sum_res[col_block_start_idx: col_block_stop_idx] += np.sum(block, axis=0)

        row_count += row_block_size
        _verbose_print(
            "Processed " + str(min(len(rows_to_load), row_count)) + " of " + str(len(rows_to_load)) + " rows.")

    return sum_res


class SetCoveringMachine(BaseSetCoveringMachine):
    """
    The Set Covering Machine (SCM).

    Marchand, M., & Taylor, J. S. (2003). The set covering machine. Journal of Machine Learning Research, 3, 723–746.

    Parameters:
    -----------
    model_type: pyscm.model.conjunction or pyscm.model.disjunction, default=pyscm.model.conjunction
        The type of model to be built.

    p: float, default=1.0
        A parameter to control the importance of making prediction errors on positive examples in the utility function.

    max_attributes: int, default=10
        The maximum number of binary attributes to include in the model.

    verbose: bool, default=False
        Sets verbose mode on/off.
    """

    def __init__(self, model_type=conjunction, p=1.0, max_attributes=10, verbose=False):
        super(SetCoveringMachine, self).__init__(model_type=model_type, max_attributes=max_attributes, verbose=verbose)

        if model_type == conjunction:
            self.model = ConjunctionModel()
        elif model_type == disjunction:
            self.model = DisjunctionModel()
        else:
            raise ValueError("Unsupported model type.")

        self.p = p

    def _get_binary_attribute_utilities(self, attribute_classifications, positive_example_idx, negative_example_idx,
                                        n_examples, example_block_size, attribute_block_size):
        self._verbose_print("Counting covered negative examples")
        negative_cover_counts = negative_example_idx.shape[0] - _block_sum_rows(row_idx=negative_example_idx,
                                                                                array=attribute_classifications,
                                                                                n_examples=n_examples,
                                                                                row_block_size=example_block_size,
                                                                                column_block_size=attribute_block_size,
                                                                                verbose=self.verbose)
        self._verbose_print("Counting errors on positive examples")
        # It is possible that there are no more positive examples to be considered. This is not possible for negative
        # examples because of the SCM's stopping criterion.
        if positive_example_idx.shape[0] > 0:
            positive_error_counts = positive_example_idx.shape[0] - _block_sum_rows(row_idx=positive_example_idx,
                                                                                    array=attribute_classifications,
                                                                                    n_examples=n_examples,
                                                                                    row_block_size=example_block_size,
                                                                                    column_block_size=attribute_block_size,
                                                                                    verbose=self.verbose)
        else:
            positive_error_counts = np.zeros(attribute_classifications.shape[1], dtype=negative_cover_counts.dtype)

        self._verbose_print("Computing attribute utilities")
        utilities = negative_cover_counts - float(self.p) * positive_error_counts

        return utilities, positive_error_counts, negative_cover_counts


class MetaSetCoveringMachine(BaseSetCoveringMachine):
    """
    The Meta Set Covering Machine (SCM).

    Drouin et al. 2014 (Unpublished)

    Parameters:
    -----------
    model_type: pyscm.model.conjunction or pyscm.model.disjunction, default=pyscm.model.conjunction
        The type of model to be built.

    p: float, default=1.0
        A parameter to control the importance of making prediction errors on positive examples in the utility function.

    c: float, default=1.0
        A parameter to control the importance of the cardinality of the meta-attributes in the utility function.

    max_attributes: int, default=10
        The maximum number of binary attributes to include in the model.

    verbose: bool, default=False
        Sets verbose mode on/off.
    """

    def __init__(self, model_type=conjunction, p=1.0, c=1.0, max_attributes=10, verbose=False):
        super(MetaSetCoveringMachine, self).__init__(model_type=model_type, max_attributes=max_attributes,
                                                     verbose=verbose)

        if model_type == conjunction:
            self.model = ConjunctionModel()
        elif model_type == disjunction:
            self.model = DisjunctionModel()
        else:
            raise ValueError("Unsupported model type.")

        self.p = p
        self.c = c

        self._flags["PROBABILISTIC_PREDICTIONS"] = True

    def fit(self, binary_attributes, y, X=None, meta_attribute_classifications=None, model_append_callback=None,
            example_block_size=64, attribute_block_size=1000):
        """
        """
        if X is None and meta_attribute_classifications is None:
            raise ValueError("X or meta_attribute_classifications must have a value.")

        if meta_attribute_classifications is None:
            self._verbose_print("Got " + str(len(binary_attributes)) + " binary attributes.")
            if not isinstance(binary_attributes, SingleBinaryAttributeList):
                raise ValueError("The provided binary attribute list must contain single binary attributes only if" + \
                                 " the attribute classifications are not precomputed.")
            self._verbose_print("Classifying the examples with the binary attributes")
            meta_attribute_classifications = binary_attributes.classify(X)
            # TODO: Compress into meta attributes
            raise NotImplementedError()

        else:
            if not isinstance(binary_attributes, MetaBinaryAttributeList):
                raise ValueError("The provided binary attribute list must contain binary meta-attributes only if" + \
                                 " the meta-attribute classifications are precomputed.")

            self._verbose_print("Binary attribute classifications were precomputed")
            if meta_attribute_classifications.shape[1] != len(binary_attributes):
                raise ValueError("The number of attributes must match in attribute_classifications and",
                                 "binary_attributes.")

        super(MetaSetCoveringMachine, self).fit(binary_attributes=binary_attributes,
                                                y=y,
                                                X=X,
                                                attribute_classifications=meta_attribute_classifications,
                                                model_append_callback=model_append_callback,
                                                example_block_size=example_block_size,
                                                attribute_block_size=attribute_block_size,
                                                utility__meta_attribute_cardinalities=np.log(
                                                    binary_attributes.cardinalities))

    def _get_binary_attribute_utilities(self, attribute_classifications, positive_example_idx, negative_example_idx,
                                        n_examples, example_block_size, attribute_block_size,
                                        meta_attribute_cardinalities):
        self._verbose_print("Counting covered negative examples")
        negative_cover_counts = negative_example_idx.shape[0] - _block_sum_rows(row_idx=negative_example_idx,
                                                                                array=attribute_classifications,
                                                                                n_examples=n_examples,
                                                                                row_block_size=example_block_size,
                                                                                column_block_size=attribute_block_size,
                                                                                verbose=self.verbose)

        self._verbose_print("Counting errors on positive examples")
        # It is possible that there are no more positive examples to be considered. This is not possible for negative
        # examples because of the SCM's stopping criterion.
        if positive_example_idx.shape[0] > 0:
            positive_error_counts = positive_example_idx.shape[0] - _block_sum_rows(row_idx=positive_example_idx,
                                                                                    array=attribute_classifications,
                                                                                    n_example=n_examples,
                                                                                    row_block_size=example_block_size,
                                                                                    column_block_size=attribute_block_size,
                                                                                    verbose=self.verbose)
        else:
            positive_error_counts = np.zeros(attribute_classifications.shape[1], dtype=negative_cover_counts.dtype)

        self._verbose_print("Computing attribute utilities")
        utilities = negative_cover_counts - float(self.p) * positive_error_counts + float(
            self.c) * meta_attribute_cardinalities

        return utilities, positive_error_counts, negative_cover_counts

    def predict_proba(self, X):
        """
        Compute probabilistic predictions.

        Parameters:
        -----------
        X: numpy_array, shape=(n_examples,)
            The feature vectors associated to some examples.

        Returns:
        --------
        predictions: numpy_array, shape=(n_examples,)
            The probability of each example belonging to the positive class. The positive class is the greatest class
            based on the sorted class labels. e.g.: For classes -1 and +1, -1 is the negative class and +1 is the
            positive class.
        """
        return self._predict_proba(X)
