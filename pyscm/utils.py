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
from numpy import sort

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
        [str(k) + "=" + str(v) for k, v in instance.__dict__.iteritems() if str(k[0]) != "_"]) + ")"


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
        print text


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
    a_list = sort(a_list, kind="heapsort")

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
