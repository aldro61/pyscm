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
from ..utils import _class_to_string


class ModelMixin:
    def __init__(self):
        self.binary_attributes = []
        self._example_dependencies = []

    def add(self, binary_attribute):
        self.binary_attributes.append(binary_attribute)

        if len(binary_attribute.example_dependencies) > 0:
            self._example_dependencies += binary_attribute.example_dependencies

    def remove(self, index):
        del self.binary_attributes[index]

    @property
    def example_dependencies(self):
        return list(self._example_dependencies)

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.binary_attributes])

    def __len__(self):
        return len(self.binary_attributes)

    def __str__(self):
        return self._to_string()