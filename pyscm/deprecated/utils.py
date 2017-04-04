#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from six import iteritems
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

def _chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Parameters:
    -----------
    l: list like container
        Any instance that implements __getitem__.

    n: int
        The size of the chunks.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return instance.__class__.__name__ + "(" + ",".join(
        [str(k) + "=" + str(v) for k, v in itetitems(instance.__dict__) if str(k[0]) != "_"]) + ")"

def _conditional_print(text, condition):
    """
    Prints text if the condition is True.

    Parameters:
    -----------
    text: string
        The text to print

    condition: bool
        Print or do not print
    """
    if condition:
        print(text)

def _split_into_contiguous(a_list):
    """
    Returns a list of contiguous sublists

    Parameters:
    -----------
    a_list: list
        A list of elements. These elements must implement __add__ and __eq__

    Returns:
    --------
    contiguous_blocks: list of lists
        A list containing contiguous sublists of a_list
    """
    a_list = np.sort(a_list, kind="heapsort")

    split = []
    contiguous = []
    for j in a_list:
        if contiguous == []:
            contiguous.append(j)
        else:
            if contiguous[-1]+1 == j:
                contiguous.append(j)
            else:
                split.append(contiguous)
                contiguous = [j]
    split.append(contiguous)

    return split

def _pack_binary_bytes_to_ints(a, int_size):
    """
    Packs binary values stored in bytes into ints
    """
    pack_size = int_size
    if pack_size == 64:
        type = np.uint64
    elif pack_size == 32:
        type = np.uint32
    else:
        raise ValueError("Supported data types are 32-bit and 64-bit integers.")

    b = np.zeros((int(ceil(1.0 * a.shape[0] / pack_size)), a.shape[1]), dtype=type)

    packed_rows = 0
    packing_row = 0
    for i in xrange(a.shape[0]):
        if packed_rows == pack_size:
            packed_rows = 0
            packing_row += 1
        tmp = np.asarray(a[i], dtype=type)
        tmp = np.left_shift(tmp, type(pack_size - packed_rows - 1))
        np.bitwise_or(b[packing_row], tmp, out=b[packing_row])
        packed_rows += 1

    return b

def _unpack_binary_bytes_from_ints(a):
    """
    Unpacks binary values stored in bytes into ints
    """
    type = a.dtype

    if type == np.uint32:
        pack_size = 32
    elif type == np.uint64:
        pack_size = 64
    else:
        raise ValueError("Supported data types are 32-bit and 64-bit integers.")

    unpacked_n_rows = a.shape[0] * pack_size
    unpacked_n_columns = a.shape[1] if len(a.shape) > 1 else 1
    b = np.zeros((unpacked_n_rows, a.shape[1]) if len(a.shape) > 1 else unpacked_n_rows, dtype=np.uint8)

    packed_rows = 0
    packing_row = 0
    for i in xrange(b.shape[0]):
        if packed_rows == pack_size:
            packed_rows = 0
            packing_row += 1
        tmp = np.left_shift(np.ones(unpacked_n_columns, dtype=type), pack_size - (i - pack_size * packing_row)-1)
        np.bitwise_and(a[packing_row], tmp, tmp)
        b[i] = tmp > 0
        packed_rows += 1

    return b
