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
try:
    import h5py
except:
    h5py = None


class H5pyDataset(object):
    """
    A proxy class for a H5Py dataset object.

    Parameters:
    -----------
    file_name: string
        The name of the HDF5 file containing the dataset

    dataset_path: string
        The path to the dataset in the file.

    Note:
    -----
    This proxy class must be used to used precomputed binary classifications stored in an HDF5 file. It allows using
    multiprocessing even though HDF5 is not fully compatible with the multiprocessing package (See below).

    Importantly, only one process actually reads/writes the HDF5 file. Remember that when a process is fork()ed, the
    child inherits the HDF5 state from its parent, which can be dangerous if you already have a file open. Trying to
    interact with the same file on disk from multiple processes results in undefined behavior.
    """

    def __init__(self, file_name, dataset_path):
        self.file_name = file_name
        self.dataset_path = dataset_path

        #TODO: Ensure that the file is not open somewhere in memory

    @property
    def shape(self):
        dataset = h5py.File(name=self.file_name, mode="r")[self.dataset_path]
        shape = dataset.shape
        dataset.file.close()
        return shape

    def __getitem__(self, slice):
        dataset = h5py.File(name=self.file_name, mode="r")[self.dataset_path]
        data = dataset[slice]
        dataset.file.close()
        return data

    def __len__(self):
        dataset = h5py.File(name=self.file_name, mode="r")[self.dataset_path]
        length = len(dataset)
        dataset.file.close()
        return length