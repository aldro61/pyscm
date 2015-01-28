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
import h5py as h
import numpy as np
from math import ceil
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

class HDF5PackedAttributeClassifications(BaseAttributeClassifications):
    def __init__(self, datasets, n_examples, row_block_size=None, col_block_size=None):
        self.datasets = datasets
        self.dataset_n_examples = n_examples
        self.total_n_examples = sum(self.dataset_n_examples)
        self.dataset_n_cols = self.datasets[0].shape[1]
        self.dataset_dtype = self.datasets[0].dtype

        if row_block_size is None:
            self.row_block_size = self.datasets[0].chunks[0]
        if col_block_size is None:
            self.col_block_size = self.datasets[0].chunks[1]

        for dataset in self.datasets[1:]:
            if dataset.shape[1] != self.dataset_n_cols:
                raise RuntimeError("All datasets must have the same number of columns.")
            if dataset.dtype != self.dataset_dtype:
                raise RuntimeError("All datasets must have the same data type.")

        self.dataset_stop_example = [0] * len(self.datasets)
        for i, dataset in enumerate(self.datasets):
            if i == 0:
                previous_dataset_stop = 0
            else:
                previous_dataset_stop = self.dataset_stop_example[i - 1]
            self.dataset_stop_example[i] = previous_dataset_stop + self.dataset_n_examples[i]

        self.dataset_start_example = [0] * len(self.datasets)
        for i, dataset in enumerate(self.datasets):
            if i == 0:
                self.dataset_start_example[i] = 0
            else:
                self.dataset_start_example[i] = self.dataset_stop_example[i - 1]

        # Get the size of the ints used to store the data
        if self.datasets[0].dtype == np.uint32:
            self.dataset_pack_size = 32
            self.inplace_popcount = inplace_popcount_32
        elif self.datasets[0].dtype == np.uint64:
            self.dataset_pack_size = 64
            self.inplace_popcount = inplace_popcount_64
        else:
            raise ValueError("Unsupported data type for packed attribute classifications array. The supported data" +
                             " types are np.uint32 and np.uint64.")

        super(BaseAttributeClassifications, self).__init__()

    def get_column(self, idx):
        result = np.zeros(self.total_n_examples, dtype=np.uint8)
        for i, dataset in enumerate(self.datasets):
            result[self.dataset_start_example[i]:self.dataset_stop_example[i]] = \
                _unpack_binary_bytes_from_ints(dataset[:, idx])[: self.dataset_n_examples[i]]
        return result

    @property
    def shape(self):
        return self.total_n_examples, self.dataset_n_cols

    def sum_rows(self, rows):
        rows = np.asarray(rows)
        result_dtype = _column_sum_dtype(rows)
        result = np.zeros(self.dataset_n_cols, dtype=result_dtype)

        # Builds a mask to turn off the bits of the rows we do not want to count in the sum.
        #TODO: this could be in utils as build int mask, example_idx could be set_bit_idx
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

        # Find the rows that occur in each dataset and their relative index
        # XXX: This could be faster if a binary search was used.
        rows = np.sort(rows)
        dataset_relative_rows = [[] for _ in xrange(len(self.datasets))]
        for row_idx in rows:
            for i in xrange(len(self.datasets)):
                if row_idx >= self.dataset_start_example[i] and row_idx < self.dataset_stop_example[i]:
                    dataset_relative_rows[i].append(row_idx - self.dataset_start_example[i])
                    break

        # Create a row mask for each dataset
        dataset_row_masks = [build_row_mask(dataset_relative_rows[i],
                                            self.dataset_n_examples[i],
                                            self.dataset_pack_size)
                             if len(dataset_relative_rows[i]) > 0 else []
                             for i in xrange(len(self.datasets))]
        del dataset_relative_rows

        # For each dataset load the rows for which the mask is not 0. Support column slicing aswell
        n_col_blocks = int(ceil(1.0 * self.dataset_n_cols / self.col_block_size))
        for i, dataset in enumerate(self.datasets):
            row_mask = dataset_row_masks[i]

            if len(row_mask) == 0:
                # print "Dont need to load anything from", i+1
                # print
                continue

            rows_to_load = np.where(row_mask != 0)[0]
            # print "The row masks are:", row_mask
            # print "We must only load rows:", rows_to_load
            # print "Their masks are:", row_mask[rows_to_load]

            n_row_blocks = int(ceil(1.0 * len(rows_to_load) / self.row_block_size))

            for row_block in xrange(n_row_blocks):
                for col_block in xrange(n_col_blocks):

                    # Load the appropriate rows/columns based on the block sizes
                    block = dataset[rows_to_load[row_block * self.row_block_size:(row_block + 1) * self.row_block_size],
                            col_block * self.col_block_size:(col_block + 1)*self.col_block_size]

                    # Popcount
                    if len(block.shape) == 1:
                        block = block.reshape(1, -1)
                    self.inplace_popcount(block, row_mask)

                    # Increment the sum
                    result[col_block * self.col_block_size:(col_block + 1) * self.col_block_size] += np.sum(block, axis=0)

        return result

#TODO: Support unpacked learning from HDF5
class HDF5UnpackedAttributeClassifications(BaseAttributeClassifications):
    pass