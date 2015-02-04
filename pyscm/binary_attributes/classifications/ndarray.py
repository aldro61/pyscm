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
from math import ceil
#TODO MAKE THESE IMPORTS RELATIVE
from .base import BaseAttributeClassifications
from .popcount import inplace_popcount_32, inplace_popcount_64
from ...utils import _unpack_binary_bytes_from_ints


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

class NumpyPackedAttributeClassifications(BaseAttributeClassifications):
    def __init__(self, array, n_rows):
        self.array = array
        self.n_examples = n_rows

        #TODO: The numpy array is already in RAM. Create a cython function that sums without replacing the values in the
        # received array. By doing so, we will be able to remove the block size parameters and process the entire array
        # without additionnal memory usage.
        self.row_block_size = 1
        self.col_block_size = 1000
        super(BaseAttributeClassifications, self).__init__()

    def get_column(self, column):
        return _unpack_binary_bytes_from_ints(self.array[:, column])[: self.n_examples]

    @property
    def shape(self):
        return self.n_examples, self.array.shape[1]

    def sum_rows(self, rows):
        # TODO: PASS A POINTER TO THE RESULTING SUM TO SAVE MEMORY
        rows = np.asarray(rows)

        # Get the size of the ints used to store the data
        if self.array.dtype == np.uint32:
            n_bits = 32
            inplace_popcount = inplace_popcount_32
        elif self.array.dtype == np.uint64:
            n_bits = 64
            inplace_popcount = inplace_popcount_64
        else:
            raise ValueError("Unsupported data type for compressed attribute classifications array. The supported data" +
                             " types are np.uint32 and np.uint64.")

        mask = build_row_mask(rows, self.n_examples, n_bits)
        rows_to_load = np.where(mask != 0)[0]  # Don't load ints if we don't need their bits.
        sum_res = np.zeros(self.array.shape[1], dtype=_column_sum_dtype(rows))

        n_row_blocks = int(ceil(float(len(rows_to_load)) / self.row_block_size))
        n_col_blocks = int(ceil(float(self.array.shape[1]) / self.col_block_size))

        row_count = 0
        for row_block in xrange(n_row_blocks):
            row_block_start_idx = row_block * self.row_block_size
            row_block_stop_idx = row_block_start_idx + self.row_block_size
            row_mask = mask[rows_to_load[row_block_start_idx: row_block_stop_idx]]

            for col_block in xrange(n_col_blocks):
                col_block_start_idx = col_block * self.col_block_size
                col_block_stop_idx = col_block_start_idx + self.col_block_size
                block = self.array[rows_to_load[row_block_start_idx: row_block_stop_idx],
                                   col_block_start_idx: col_block_stop_idx]

                if len(block.shape) == 1:
                    block = block.reshape(1, -1)

                if hasattr(block, "flags") and not block.flags["OWNDATA"]:
                    block = block.copy()

                inplace_popcount(block, row_mask)
                sum_res[col_block_start_idx: col_block_stop_idx] += np.sum(block, axis=0)

            row_count += self.row_block_size

        return sum_res

# TODO: Support unpacked learning as well
class NumpyUnpackedAttributeClassifications(BaseAttributeClassifications):
    pass