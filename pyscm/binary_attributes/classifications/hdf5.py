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
    def __init__(self, datasets, n_rows, block_size=None):
        self.datasets = datasets
        self.dataset_initial_n_rows = list(n_rows)
        self.dataset_n_rows = list(n_rows)
        self.initial_total_n_rows = sum(self.dataset_n_rows)
        self.total_n_rows = self.initial_total_n_rows
        self.dataset_n_cols = self.datasets[0].shape[1]
        self.dataset_dtype = self.datasets[0].dtype
        self.dataset_removed_rows = [[] for _ in xrange(len(self.datasets))]
        self.dataset_removed_rows_mask = [np.zeros(self.dataset_initial_n_rows[i], dtype=np.bool) for i in xrange(len(self.datasets))]
        self.block_size = (None, None)

        if block_size is None:
            if self.datasets[0].chunks is None:
                self.block_size = (1, self.datasets[0].shape[1])
            else:
                self.block_size = self.datasets[0].chunks
        else:
            if len(block_size) != 2 or not isinstance(block_size[0], int) or not isinstance(block_size[1], int):
                raise ValueError("The block size must be a tuple of 2 integers.")
            self.block_size = block_size

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
            self.dataset_stop_example[i] = previous_dataset_stop + self.dataset_n_rows[i]

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

    def get_columns(self, columns):
        result = np.zeros((self.total_n_rows,) if isinstance(columns, int) else (self.total_n_rows, len(columns)), dtype=np.uint8)
        for i, dataset in enumerate(self.datasets):
            row_mask = np.ones(dataset.shape[0] * self.dataset_pack_size, dtype=np.bool)
            row_mask[self.dataset_initial_n_rows[i] : ] = False
            row_mask[self.dataset_removed_rows[i]] = False
            result[self.dataset_start_example[i]:self.dataset_stop_example[i]] = \
                _unpack_binary_bytes_from_ints(dataset[:, columns])[row_mask]
        return result

    def remove_rows(self, rows):
        # Find in which dataset the rows must be removed
        dataset_removed_rows = [[] for _ in xrange(len(self.datasets))]
        for row_idx in rows:
            ds_idx = self._get_row_dataset(row_idx)
            if ds_idx == -1:
                raise IndexError("Row index %d is out of bounds for array of shape (%d, %d)" % (row_idx, self.shape[0],
                                                                                                self.shape[1]))
            # Find which row in the dataset corresponds to the element to delete
            # TODO: This is inefficient!
            relative_row_index = row_idx - self.dataset_start_example[ds_idx]
            current_idx = -1
            n_active_elements_seen = 0
            while n_active_elements_seen < relative_row_index + 1:
                current_idx += 1
                if not self.dataset_removed_rows_mask[ds_idx][current_idx]:
                    n_active_elements_seen += 1

            dataset_removed_rows[ds_idx].append(current_idx)
        # Update the dataset removed row lists
        # Update the start and stop indexes
        # Adjust the shape
        # Adjust the number of rows in each dataset
        # Store the sorted relative removed row indexes by dataset
        for i in xrange(len(self.datasets)):
            if len(dataset_removed_rows[i]) > 0:
                self.dataset_removed_rows[i] = sorted(set(self.dataset_removed_rows[i] + dataset_removed_rows[i]))
                self.dataset_removed_rows_mask[i] = np.zeros(self.dataset_initial_n_rows[i], dtype=np.bool)
                self.dataset_removed_rows_mask[i][self.dataset_removed_rows[i]] = True
                self.dataset_n_rows[i] = self.dataset_initial_n_rows[i] - len(self.dataset_removed_rows[i])
            self.dataset_stop_example[i] = self.dataset_n_rows[i] + (0 if i == 0 else self.dataset_stop_example[i - 1])
            self.dataset_start_example[i] = 0 if i == 0 else self.dataset_stop_example[i - 1]
        self.total_n_rows = sum(self.dataset_n_rows)

    @property
    def shape(self):
        return self.total_n_rows, self.dataset_n_cols

    def sum_rows(self, rows):
        """
        Note: Assumes that rows does not contain duplicate elements. Rows will not be considered more than once.
        """
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
        rows = np.sort(rows)
        dataset_relative_rows = [[] for _ in xrange(len(self.datasets))]
        for row_idx in rows:
            ds_idx = self._get_row_dataset(row_idx)
            if ds_idx != -1:

                # Find which row in the dataset corresponds to the requested row
                # TODO: This is inefficient!
                relative_row_index = row_idx - self.dataset_start_example[ds_idx]
                current_idx = -1
                n_active_elements_seen = 0
                while n_active_elements_seen < relative_row_index + 1:
                    current_idx += 1
                    if not self.dataset_removed_rows_mask[ds_idx][current_idx]:
                        n_active_elements_seen += 1

                dataset_relative_rows[ds_idx].append(current_idx)
            else:
                raise IndexError("Row index %d is out of bounds for array of shape (%d, %d)" % (row_idx, self.shape[0],
                                                                                                self.shape[1]))
        # Create a row mask for each dataset
        dataset_row_masks = [build_row_mask(dataset_relative_rows[i],
                                            self.dataset_initial_n_rows[i],
                                            self.dataset_pack_size)
                             if len(dataset_relative_rows[i]) > 0 else []
                             for i in xrange(len(self.datasets))]
        del dataset_relative_rows

        # For each dataset load the rows for which the mask is not 0. Support column slicing aswell
        n_col_blocks = int(ceil(1.0 * self.dataset_n_cols / self.block_size[1]))
        for i, dataset in enumerate(self.datasets):
            row_mask = dataset_row_masks[i]

            # TODO: Could this be true? I don't think so.
            if len(row_mask) == 0:
                continue

            rows_to_load = np.where(row_mask != 0)[0]

            n_row_blocks = int(ceil(1.0 * len(rows_to_load) / self.block_size[0]))

            for row_block in xrange(n_row_blocks):
                block_row_mask = row_mask[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]]]

                for col_block in xrange(n_col_blocks):

                    # Load the appropriate rows/columns based on the block sizes
                    block = dataset[rows_to_load[row_block * self.block_size[0]:(row_block + 1) * self.block_size[0]],
                            col_block * self.block_size[1]:(col_block + 1)*self.block_size[1]]

                    # Popcount
                    if len(block.shape) == 1:
                        block = block.reshape(1, -1)
                    self.inplace_popcount(block, block_row_mask)

                    # Increment the sum
                    result[col_block * self.block_size[1]:(col_block + 1) * self.block_size[1]] += np.sum(block, axis=0)

        return result

    def _get_row_dataset(self, row_idx):
        # TODO: This could be faster if we used a binary search
        for i in xrange(len(self.datasets)):
                if self.dataset_stop_example[i] > row_idx >= self.dataset_start_example[i]:
                    return i
        return -1

#TODO: Support unpacked learning from HDF5
class HDF5UnpackedAttributeClassifications(BaseAttributeClassifications):
    pass